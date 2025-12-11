import warnings
warnings.filterwarnings('ignore')

import argparse
import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

# Import hcmm model parameters
import Model as hcmm


# ------------------- Data Loading/Cleaning/Scaling Functions ------------------- #
def get_data(model_data, raw_data):
    """
    Load the model and cleaned raw meteor datasets.

    Parameters
    ----------
    model_data : str
        File path to the processed (model) CSV file.
    raw_data : str
        File path to the raw meteor CSV file.

    Returns
    -------
    model_df : pandas.DataFrame
        Loaded model dataset with stripped IAU_code values.
    raw_df : pandas.DataFrame
        Loaded raw dataset with stripped IAU_code values.
    """
    model_df = pd.read_csv(model_data, low_memory=False)
    raw_df = pd.read_csv(raw_data, low_memory=False)

    # Strip the leading spaces out of all IAU_codes
    model_df.IAU_code = model_df.IAU_code.str.strip()
    raw_df.IAU_code = raw_df.IAU_code.str.strip()

    return model_df, raw_df


def scale_data(model_data, raw_data):
    """
    Scale selected features, remove unphysical values, and
    align the raw data to the cleaned model dataset.

    Parameters
    ----------
    model_data : pandas.DataFrame
        Processed meteor dataset to be cleaned and log-transformed.
    raw_data : pandas.DataFrame
        Raw meteor dataset used for trajectory alignment.

    Returns
    -------
    scaled_df : pandas.DataFrame
        Cleaned and log-transformed model data with a 'Group' label.
    matched_raw : pandas.DataFrame
        Raw data filtered to only include trajectories present in scaled_df.
    """
    # Define logarithmic features
    log_features = ['EnergyRecBeg', 'HtBeg', 'HtEnd', 'Vinit', 'Vavg',
                    'Deceleration', 'AtmDensBeg','TrailLength', 'Mass_kg']

    # Make a copy to preserve the original
    scaled_df = model_data.copy()

    # Drop rows with unphysical values
    scaled_df = scaled_df[scaled_df['Deceleration'] >= 0]
    scaled_df = scaled_df[scaled_df['Mass_kg'] > 0]
    scaled_df = scaled_df[scaled_df['F'] >= 0]
    scaled_df = scaled_df[scaled_df['AbsMag'] > -10]
    scaled_df = scaled_df[scaled_df['EnergyRecBeg'] >= 0]
    scaled_df = scaled_df[scaled_df['Kb'].notna()]
    scaled_df.drop(columns='Elev')

    # Apply log10 only to specified features (e.g., 'EnergyRec', 'Vinit', etc.)
    scaled_df[log_features] = np.log10(scaled_df[log_features])
    scaled_df.ZenithAngle = np.cos(np.radians(scaled_df.ZenithAngle))

    # Drop any NaN values
    scaled_df = scaled_df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    # Filter raw_data to keep only trajectories that exist in meteor_df
    valid_trajectories = set(scaled_df['Unique_trajectory'])
    matched_raw = raw_data[raw_data['Unique_trajectory'].isin(valid_trajectories)].reset_index(drop=True)

    # Assign labels to the range of Kb values
    scaled_df['Group'] = pd.cut(scaled_df['Kb'], bins=[0, 7.3, 8.0, np.inf],
                                 labels=['Cometary', 'Carbonaceous', 'Asteroidal'], right=False)

    return scaled_df, matched_raw


# ------------------- Manual Model Functions ------------------- #
def gaussian_pdf(F, mu, Sigma):
    """
    Multivariate Gaussian PDF evaluated at points F.

    Parameters
    ----------
    F : array-like, shape (m, j)
        Points at which to evaluate the PDF (factor scores).
    mu : array-like, shape (j,)
        Mean vector of the Gaussian.
    Sigma : array-like, shape (j, j)
        Covariance matrix of the Gaussian.

    Returns
    -------
    pdf : ndarray, shape (m,)
        PDF values for each row of F.
    """
    F = np.asarray(F)
    mu = np.asarray(mu)
    Sigma = np.asarray(Sigma)

    m, j = F.shape

    # Centered data: F - mu
    diff = F - mu  # (m, j)

    # Inverse and log-determinant of covariance
    Sigma_inv = np.linalg.inv(Sigma)
    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        raise ValueError("Covariance matrix is not positive definite.")

    # Quadratic form (x - mu)^T Sigma^{-1} (x - mu) for each row
    quad = np.sum(diff @ Sigma_inv * diff, axis=1)  # (m,)

    # Multivariate normal PDF
    norm_const = np.power(2 * np.pi, -j / 2) * np.exp(-0.5 * logdet)
    pdf = norm_const * np.exp(-0.5 * quad)

    return pdf


def get_cluster_params(params_df, k):
    """
    Extract mean vector and covariance matrix for cluster k from gmm_params_df.
    Assumes columns: mu_F1, mu_F2, mu_F3, Sigma_11..Sigma_33.
    """
    row = params_df.loc[k]

    mu_k = np.array([row["mu_F1"], row["mu_F2"], row["mu_F3"]])

    Sigma_k = np.array([
        [row["Sigma_11"], row["Sigma_12"], row["Sigma_13"]],
        [row["Sigma_21"], row["Sigma_22"], row["Sigma_23"]],
        [row["Sigma_31"], row["Sigma_32"], row["Sigma_33"]],
    ])

    weight_k = row["weight"]

    return mu_k, Sigma_k, weight_k


def gaussian_pdf_for_cluster_k(F, gmm_params_df, k):
    """
    Compute the Gaussian PDF N(f | mu_k, Sigma_k) for all events and cluster k.
    """
    mu_k, Sigma_k, _ = get_cluster_params(gmm_params_df, k)
    pdf_k = gaussian_pdf(F, mu_k, Sigma_k)  # shape (m,)

    return pdf_k


def compute_weighted_pdfs(F, gmm_params_df):
    """
    Compute:
      - Gaussian PDF for each cluster
      - weighted PDF for each cluster: weight_k * N(f | mu_k, Sigma_k)
      - total weighted PDF across all clusters (mixture density)

    Parameters
    ----------
    F : array-like, shape (m, j)
        Factor scores.
    gmm_params_df : DataFrame
        GMM parameters with one row per component.

    Returns
    -------
    pdfs : ndarray, shape (m, K)
        Unweighted Gaussian PDFs for each cluster.
    weighted_pdfs : ndarray, shape (m, K)
        weight_k * pdf_k(f_m) for each cluster.
    total_weighted_pdf : ndarray, shape (m,)
        Sum_k weight_k * pdf_k(f_m) for each event.
    """
    F = np.asarray(F)
    m, j = F.shape
    K = len(gmm_params_df)

    pdfs = np.zeros((m, K))
    weighted_pdfs = np.zeros((m, K))

    for k in range(K):
        mu_k, Sigma_k, weight_k = get_cluster_params(gmm_params_df, k)

        pdf_k = gaussian_pdf(F, mu_k, Sigma_k)        # (m,)
        pdfs[:, k] = pdf_k
        weighted_pdfs[:, k] = weight_k * pdf_k        # (m,)

    total_weighted_pdf = weighted_pdfs.sum(axis=1)    # (m,)

    return pdfs, weighted_pdfs, total_weighted_pdf


def compute_cluster_posteriors(F, gmm_params_df):
    """
    Compute posterior probabilities γ_{mk} and hard labels from F and gmm_params_df.

    Parameters
    ----------
    F : numpy.ndarray
        Feature matrix of shape (m, d) containing the transformed samples.
    gmm_params_df : pandas.DataFrame
        DataFrame containing GMM parameters (weights, means, covariances).

    Returns
    -------
    gammas : numpy.ndarray
        Posterior probabilities γ_{mk} for each sample and cluster.
    labels : numpy.ndarray
        Hard cluster assignments obtained by argmax over posterior probabilities.
    """
    pdfs, weighted_pdfs, total_weighted_pdf = compute_weighted_pdfs(F, gmm_params_df)

    # Avoid division by zero
    eps = 1e-15
    denom = total_weighted_pdf[:, None] + eps  # shape (m, 1)

    gammas = weighted_pdfs / denom
    labels = np.argmax(gammas, axis=1)

    return gammas, labels


def normalize(model, meteors):
    """
    Apply the model’s min–max normalization to a meteor feature DataFrame.

    Parameters
    ----------
    model : object instance
        Model instance containing `normalization_min` and `normalization_max`
        arrays aligned with the columns of `meteors`.
    meteors : pandas.DataFrame
        DataFrame of meteor features to normalize.

    Returns
    -------
    scaled : pandas.DataFrame
        Meteors scaled to the range [-1, 1] using the model’s normalization
        coefficients.
    """
    # Convert to DataFrame so alignment with columns is guaranteed
    min_df = pd.Series(model.normalization_min, index=meteors.columns)
    max_df = pd.Series(model.normalization_max, index=meteors.columns)

    # Min–max scale to [-1, 1]
    D = 2 / (max_df - min_df)
    b = -1 - D * min_df.T
    scaled = D * meteors + b

    return scaled


def factor_analysis(model, normalized_meteors):
    """
    Compute factor scores using the model's factor analysis parameters.

    Parameters
    ----------
    model : object instance
        Model containing FA means, noise terms, and component loadings.
    normalized_meteors : pandas.DataFrame
        Normalized meteor features to project into factor space.

    Returns
    -------
    f_hat : numpy.ndarray
        Estimated factor scores for each meteor.
    """
    # Create a dataframe of FA coefficients
    fa_params_df = pd.DataFrame({
        "feature": model.features,
        "mean": model.fa_means,
        "psi": model.fa_noise,
        "W_1": model.fa_components[0, :],
        "W_2": model.fa_components[1, :],
        "W_3": model.fa_components[2, :],
    }).set_index("feature")

    # Number of factors
    j = 3

    # Extract FA params
    mu = fa_params_df["mean"].values
    psi = fa_params_df["psi"].values
    Lambda = fa_params_df[["W_1", "W_2", "W_3"]].values

    # Build Psi^-1
    Psi_inv = np.diag(1.0 / psi)

    # Construct Omega
    Lambda_T = Lambda.T
    I_j = np.eye(j)
    omega = np.linalg.inv(Lambda_T @ Psi_inv @ Lambda + I_j) @ (Lambda_T @ Psi_inv)

    # Center
    x_centered = normalized_meteors - mu

    # Factor scores
    f_hat = x_centered @ omega.T

    return f_hat


def meteor_gaussian_mixture(model, fa_meteors):
    """
    Compute GMM posterior probabilities and cluster labels for factor scores.

    Parameters
    ----------
    model : object instance
        Model containing GMM weights, means, and covariance matrices.
    fa_meteors : numpy.ndarray
        Factor-analysis scores for each meteor.

    Returns
    -------
    gammas : numpy.ndarray
        Posterior probabilities for each meteor and each GMM component.
    labels : numpy.ndarray
        Hard cluster assignments based on maximum posterior probability.
    """
    rows = []
    for k in range(len(model.gmm_weights)):
        sigma = model.gmm_covariances[k]
        rows.append({
            "component": k,
            "weight": model.gmm_weights[k],
            "mu_F1": model.gmm_means[k, 0],
            "mu_F2": model.gmm_means[k, 1],
            "mu_F3": model.gmm_means[k, 2],
            "Sigma_11": sigma[0, 0],
            "Sigma_12": sigma[0, 1],
            "Sigma_13": sigma[0, 2],
            "Sigma_21": sigma[1, 0],
            "Sigma_22": sigma[1, 1],
            "Sigma_23": sigma[1, 2],
            "Sigma_31": sigma[2, 0],
            "Sigma_32": sigma[2, 1],
            "Sigma_33": sigma[2, 2],
        })

    gmm_params_df = pd.DataFrame(rows).set_index("component")
    gammas, labels = compute_cluster_posteriors(fa_meteors, gmm_params_df)

    return gammas, labels


# ------------------- Summary/Plotting Functions ------------------- #
def classification_summary(meteors, class_assignments, probabilities, legend_labels, conf_thresh):
        """
        Assign clusters, apply confidence filtering, and summarize counts per shower
        before and after filtering, including cluster distributions.
        """
        # Assign cluster metadata
        meteors_labeled = meteors.copy()
        meteors_labeled["Cluster"] = class_assignments
        meteors_labeled["MaxProb"] = probabilities.max(axis=1)
        meteors_labeled["Cluster_Label"] = meteors_labeled["Cluster"].map(legend_labels)

        # Pre-filter counts
        pre_counts = meteors_labeled["IAU_code"].value_counts().rename("Pre_Count")
        total_pre = int(pre_counts.sum())
        print(f"Before filtering: {total_pre:,} events")

        # Apply confidence threshold
        filtered = meteors_labeled[meteors_labeled["MaxProb"] >= conf_thresh]

        # Post-filter counts
        post_counts = filtered["IAU_code"].value_counts().rename("Post_Count")
        total_post = int(post_counts.sum())
        print(f"Remaining after filtering: {total_post:,} events")

        # Combine pre/post counts
        retention = (
            pd.concat([pre_counts, post_counts], axis=1)
            .fillna(0)
            .astype({"Pre_Count": int, "Post_Count": int})
        )
        retention["Retention_%"] = (100 * retention["Post_Count"] / retention["Pre_Count"]).round(1)

        # Cluster counts per shower
        cluster_counts = (
            filtered.groupby(["IAU_code", "Cluster_Label"])
            .size()
            .unstack(fill_value=0)
        )

        # Cluster percentages
        cluster_pct = (cluster_counts.div(cluster_counts.sum(axis=1), axis=0) * 100).round(1)
        cluster_pct.columns = [f"{col}_%" for col in cluster_pct.columns]

        # Combine counts + percentages
        cluster_summary = pd.concat([cluster_counts, cluster_pct], axis=1)

        # Merge with retention info
        retention = retention.merge(cluster_summary, left_index=True, right_index=True, how="left")

        # Build totals row
        total_counts = cluster_counts.sum()
        total_pct = (total_counts / total_post * 100).round(1)

        totals = {
            "IAU_code": "Total",
            "Pre_Count": total_pre,
            "Post_Count": total_post,
            "Retention_%": round(100 * total_post / total_pre, 1),
            **total_counts.to_dict(),
            **{f"{k}_%": v for k, v in total_pct.to_dict().items()}
        }

        retention = retention.reset_index().sort_values("Retention_%", ascending=False)
        retention = pd.concat([retention, pd.DataFrame([totals])], ignore_index=True)

        print("\nPost-filtering summary:\n", retention.fillna(0).to_string(index=False))

        if args.save_output:
            retention.to_csv(os.path.join(args.save_path, "classification_summary.csv"), index=False)

        return filtered, retention


def plot_class_pie(retention_summary, legend_labels):
    """
    Plot a donut chart of total meteors per class.
    """
    # Extract Total row
    total_row = retention_summary[retention_summary["IAU_code"] == "Total"]

    # Class columns in the order defined by legend_labels
    class_cols = list(legend_labels.values())
    counts = total_row[class_cols].iloc[0].values
    labels = class_cols

    # Plot donut chart
    fig, ax = plt.subplots(figsize=(6, 6))

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=labels,
        colors=palette,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.80,
        labeldistance=1.05,
        wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'},
        textprops={'fontsize': 12},
    )

    # Draw donut hole
    centre_circle = plt.Circle((0, 0), 0.60, fc='white')
    ax.add_artist(centre_circle)

    # Add total count in center
    total_count = int(counts.sum())
    ax.text(0, 0, f"{total_count:,}\nmeteors",
            ha='center', va='center', fontsize=16, weight='bold')

    ax.set_title("$H_{class}$ Distribution", fontsize=16)
    ax.axis('equal')  # keeps chart circular

    plt.tight_layout()
    if args.save_output:
        plt.savefig(os.path.join(args.save_path, "classification_distribution.jpg"), dpi=600)
    plt.show()


if __name__ == '__main__':
    # Set parameters for Pandas
    pd.DataFrame.iteritems = pd.DataFrame.items

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_classes", type=int, required=True,
                        help="Number of classes (3 or 11)")
    parser.add_argument("-modeldata", type=str, required=True,
                        help="Filepath for .csv of data to use for classification")
    parser.add_argument("-rawdata", type=str, required=True,
                        help="Filepath for .csv of cleaned GMN data produced by gmnDataConverter.py")
    parser.add_argument("-save_output", action="store_true", default=False,
                        help="Save classification results (Summary, pie chart, and event classification")
    parser.add_argument("-threshold", type=float, default=0.00,
                        help="Probability threshold for confident assignment (0.00 <= threshold <= 1.00, default: 0.00)")
    parser.add_argument("-save_path", type=str, default=None,
                        help="Save path for plots and summary output (default: modeldata directory)")
    args = parser.parse_args()

    # If save_output and save_path not given, use directory of modeldata
    if args.save_output and args.save_path is None:
        args.save_path = os.path.dirname(args.modeldata)

    # Global color palette for all plots
    global_color_palette = cc.glasbey_category10
    palette = sns.color_palette(global_color_palette, n_colors=args.n_classes)

    # Instantiate the model
    classifier = hcmm.HClassModel(n_clusters=args.n_classes)

    # Load dataframes with meteors (observed parameters calculated from gmn data)
    # and the cleaned, raw gmn data
    load_meteors, load_raw = get_data(args.modeldata, args.rawdata)

    # Scale the data and make sure the same trajectories are in both dataframes
    meteors, raw = scale_data(load_meteors, load_raw)

    # Transform data using the model's normalization and factor analysis schemes
    X_norm = normalize(classifier, meteors[classifier.features])
    X_fa = factor_analysis(classifier, X_norm)

    # Predict clusters for new data
    class_probs, class_labels = meteor_gaussian_mixture(classifier, X_fa)

    if args.threshold != 0.00:
        print(f"Applying confidence threshold: {args.threshold:.2f}")

    meteors_labeled, retention_summary = classification_summary(
        meteors=meteors,
        class_assignments=class_labels,
        probabilities=class_probs,
        legend_labels=classifier.cluster_labels,
        conf_thresh=args.threshold
    )

    plot_class_pie(retention_summary, classifier.cluster_labels)

    if args.save_output:
        meteors_labeled[["Unique_trajectory", "Cluster_Label"]].to_csv(
            os.path.join(args.save_path, "classified_events.csv"), index=False)