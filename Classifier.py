import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import plotly.io as pio
import numpy as np
import os
import seaborn as sns
import colorcet as cc
import Model as hc

pio.renderers.default = "browser"

# ------------------- Data Loading/Cleaning/Scaling Functions ------------------- #
def clean_data(derived_data, raw_data):
    # filter out some of the bad rows such as:
    #   -EnergyRec < 0
    #   -Kb is NaN
    bad_rows = derived_data[(derived_data.EnergyRecBeg < 0) | (derived_data.Kb.isna())]
    bad_trajectories = bad_rows.Unique_trajectory.unique()
    derived_data = derived_data[~derived_data.Unique_trajectory.isin(bad_trajectories)]

    # Filter the trajectories with -inf mass
    bad_masses = derived_data.loc[np.isneginf(derived_data.Mass_kg), 'Unique_trajectory'].unique()

    # Remove them from derived_data
    derived_data = derived_data[~derived_data['Unique_trajectory'].isin(bad_masses)].reset_index(drop=True)

    # Sort both DataFrames by Unique_trajectory
    raw_data = raw_data.sort_values('Unique_trajectory').reset_index(drop=True)
    derived_data = derived_data.sort_values('Unique_trajectory').reset_index(drop=True)

    # Assign labels to the range of Kb values
    derived_data['Group'] = pd.cut(derived_data['Kb'], bins=[0, 7.3, 8.0, np.inf],
                                   labels=['Cometary', 'Carbonaceous', 'Asteroidal'], right=False)

    return derived_data, raw_data


def get_data(model_data, raw_data):
    model_df = pd.read_csv(model_data, low_memory=False)
    raw_df = pd.read_csv(raw_data, low_memory=False)

    # Strip the leading spaces out of all IAU_codes
    model_df.IAU_code = model_df.IAU_code.str.strip()
    raw_df.IAU_code = raw_df.IAU_code.str.strip()

    return model_df, raw_df


def scale_data(clean_calc, clean_raw):
    # Make a copy to preserve the original
    scaled_df = clean_calc.copy()

    # Define logarithmic features
    log_features = ['EnergyRecBeg', 'HtBeg', 'HtEnd', 'Vinit', 'Vavg',
                    'Deceleration', 'AtmDensBeg','TrailLength', 'Mass_kg']

    # Drop rows with unphysical values
    scaled_df = scaled_df[scaled_df.Deceleration >= 0]
    scaled_df = scaled_df[scaled_df.Mass_kg > 0]
    scaled_df = scaled_df[scaled_df.F >= 0]
    scaled_df = scaled_df[scaled_df.AbsMag > -10]
    scaled_df.drop(columns='Elev')

    # Apply log10 only to specified features (e.g., 'EnergyRec', 'Vinit', etc.)
    scaled_df[log_features] = np.log10(scaled_df[log_features])
    scaled_df.ZenithAngle = np.cos(np.radians(scaled_df.ZenithAngle))

    # Drop any NaN values
    scaled_df = scaled_df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    # Filter clean_raw to keep only trajectories that exist in meteor_df
    valid_trajectories = set(scaled_df['Unique_trajectory'])
    matched_raw = clean_raw[clean_raw['Unique_trajectory'].isin(valid_trajectories)].reset_index(drop=True)

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
    """
    pdfs, weighted_pdfs, total_weighted_pdf = compute_weighted_pdfs(F, gmm_params_df)

    # Avoid division by zero
    eps = 1e-15
    denom = total_weighted_pdf[:, None] + eps  # shape (m, 1)

    gammas = weighted_pdfs / denom
    labels = np.argmax(gammas, axis=1)

    return gammas, labels


def normalize(model, meteors):
    # Convert to DataFrame so alignment with columns is guaranteed
    min_df = pd.Series(model.normalization_min, index=meteors.columns)
    max_df = pd.Series(model.normalization_max, index=meteors.columns)

    # Min–max scale to [-1, 1]
    D = 2 / (max_df - min_df)
    b = -1 - D * min_df.T
    scaled = D * meteors + b

    return scaled


def factor_analysis(model, normalized_meteors):
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


# ------------------- Plotting Functions ------------------- #
def plot_sankey(flow_counts, cluster_colors, cluster_legend_labels, n_clusters, conf_thresh,
                output_dir=None):
    """Plot and optionally save a Sankey diagram for meteor shower clustering."""
    import plotly.graph_objects as go

    # Build node structure
    showers = flow_counts['IAU_code'].unique().tolist()
    clusters = flow_counts['Cluster_Label'].unique().tolist()
    all_nodes = showers + clusters
    node_indices = {name: i for i, name in enumerate(all_nodes)}

    sources = flow_counts['IAU_code'].map(node_indices)
    targets = flow_counts['Cluster_Label'].map(node_indices)
    values = flow_counts['Count']

    cluster_color_map = {label: cluster_colors[idx] for idx, label in cluster_legend_labels.items()}
    node_colors = [cluster_color_map.get(node, "lightblue") for node in all_nodes]

    # Create figure
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color="black", width=0.5),
            label=all_nodes, color=node_colors),
        link=dict(source=sources, target=targets, value=values)
    ))

    fig.update_layout(
        title_text=f"Meteor Shower Clustering for {n_clusters} Clusters",
        title_x=0.5, font=dict(size=20)
    )

    # Handle saving/showing
    if output_dir:
        fig.write_image(f"{output_dir}/sankey_fullGMN_{n_clusters}clusters_{conf_thresh}.jpg", width=1200, height=800)
        fig.write_html(f"{output_dir}/sankey_fullGMN_{n_clusters}clusters_{conf_thresh}.html")
        print(f"Sankey plots saved to {output_dir}")
    else:
        fig.show()


def summarize_filtering(train_meteors, cluster_labels, probabilities, cluster_legend_labels, conf_thresh=0.95,
                        save_path=None):
    """
    Add cluster assignments, apply a confidence filter, and summarize pre/post-filtering counts.
    Also shows per-shower distribution of clusters by counts and percentages.
    Optionally saves summary to CSV.
    """
    import pandas as pd
    import numpy as np

    # Assign clusters and metadata
    train_meteors_labeled = train_meteors.copy()
    train_meteors_labeled['Cluster'] = cluster_labels
    train_meteors_labeled['MaxProb'] = probabilities.max(axis=1)
    train_meteors_labeled['Cluster_Label'] = train_meteors_labeled['Cluster'].map(cluster_legend_labels)

    # Count pre-filter
    pre_counts = train_meteors_labeled['IAU_code'].value_counts().reset_index()
    pre_counts.columns = ['IAU_code', 'Pre_Count']
    total_pre = pre_counts['Pre_Count'].sum()
    print(f"Before filtering: {total_pre:,} events")

    # Apply confidence filter
    train_meteors_labeled = train_meteors_labeled[train_meteors_labeled['MaxProb'] >= conf_thresh]

    # Count post-filter
    post_counts = train_meteors_labeled['IAU_code'].value_counts().reset_index()
    post_counts.columns = ['IAU_code', 'Post_Count']
    total_post = post_counts['Post_Count'].sum()
    print(f"Remaining after filtering: {total_post:,} events")

    # Merge & compute retention
    retention = pre_counts.merge(post_counts, on='IAU_code', how='left').fillna(0)
    retention['Retention_%'] = (100 * retention['Post_Count'] / retention['Pre_Count']).round(1)

    # Cluster distributions per shower
    cluster_dist = (
        train_meteors_labeled.groupby(['IAU_code', 'Cluster_Label'])
        .size().unstack(fill_value=0)
    )

    # Add percentage columns for each cluster
    cluster_pct = (cluster_dist.T / cluster_dist.sum(axis=1)).T * 100
    cluster_pct = cluster_pct.round(1)
    cluster_pct.columns = [f"{col}_%" for col in cluster_pct.columns]

    # Combine counts + percentages
    cluster_summary = pd.concat([cluster_dist, cluster_pct], axis=1).reset_index()

    # Merge retention info
    retention = retention.merge(cluster_summary, on='IAU_code', how='left')

    # === Totals row with cluster totals & percentages ===
    total_counts = cluster_dist.sum(axis=0)
    total_percent = (100 * total_counts / total_post).round(1)

    total_row = {'IAU_code': 'Total',
                 'Pre_Count': total_pre,
                 'Post_Count': total_post,
                 'Retention_%': round(100 * total_post / total_pre, 1)}

    # Add cluster totals
    for col in cluster_dist.columns:
        total_row[col] = total_counts[col]
        total_row[f"{col}_%"] = total_percent[col]

    retention = pd.concat([retention.sort_values('Retention_%', ascending=False),
                           pd.DataFrame([total_row])], ignore_index=True)

    # Clean display
    print("\nPost-filtering summary:\n", retention.fillna(0).to_string(index=False))

    return train_meteors_labeled, retention


if __name__ == '__main__':
    # Set parameters for Pandas
    pd.DataFrame.iteritems = pd.DataFrame.items

    n_classes = 11
    model_data_file = "model_data_traj_summary_20251209_solrange_257.0-258.0.csv"
    raw_data_file = "cleaned_traj_summary_20251209_solrange_257.0-258.0.csv"

    # Global color palette for all plots
    global_color_palette = cc.glasbey_category10
    palette = sns.color_palette(global_color_palette, n_colors=n_classes)
    colors = [mcolors.to_hex(c) for c in palette]

    # Make a global cluster color map (by index, e.g. 0, 1, 2)
    cluster_colors = {
        k: f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"
        for k, (r, g, b) in enumerate(palette)
    }

    classifier = hc.HClassModel(n_clusters=n_classes)

    # Load dataframes with meteors (observed parameters calculated from gmn data)
    # and the cleaned, raw gmn data
    load_meteors, load_raw = get_data(model_data_file, raw_data_file)

    # Clean the loaded data (removed NaN's, add Kb groupings)
    clean_meteors, clean_raw = clean_data(load_meteors, load_raw)

    # Scale the data and make sure the same trajectories are in both dataframes
    meteors, raw = scale_data(clean_meteors, clean_raw)

    # Transform data using the model's normalization and factor analysis schemes
    X_norm = normalize(classifier, meteors[classifier.features])
    X_fa = factor_analysis(classifier, X_norm)

    # Predict clusters for new data
    probs, labels = meteor_gaussian_mixture(classifier, X_fa)

    threshold = 0.95
    print(f"Applying confidence threshold: {threshold:.2f}")
    train_meteors_labeled, retention_summary = summarize_filtering(
        train_meteors=meteors,
        cluster_labels=labels,
        probabilities=probs,
        cluster_legend_labels=classifier.cluster_labels,
        conf_thresh=threshold
    )

    flow_counts = train_meteors_labeled.groupby(['IAU_code', 'Cluster_Label']).size().reset_index(name='Count')

    # Plot the sankey diagram and save
    plot_sankey(flow_counts, cluster_colors, classifier.cluster_labels, 3, threshold)

    # # Plot the sankey diagram and DON'T save
    # plot_sankey(flow_counts, cluster_colors, cluster_legend_labels, n_clusters, threshold)