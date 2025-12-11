import numpy as np
import string
from typing import Dict
from dataclasses import dataclass

# Centralized feature dictionary
features_info = {
    'EnergyRecBeg': {
        'latex': r'$E_\mathrm{beg}$',
        'scaled_latex': r'$\log_{10}(E_\mathrm{beg})$'
    },
    'AtmDensBeg': {
        'latex': r'$\rho_\mathrm{beg}$',
        'scaled_latex': r'$\log_{10}(\rho_\mathrm{beg})$'
    },
    'HtBeg': {
        'latex': r'$Ht_\mathrm{beg}$',
        'scaled_latex': r'$\log_{10}(Ht_\mathrm{beg})$'
    },
    'HtEnd': {
        'latex': r'$Ht_\mathrm{end}$',
        'scaled_latex': r'$\log_{10}(Ht_\mathrm{end})$'
    },
    'Vinit': {
        'latex': r'$V_\mathrm{init}$',
        'scaled_latex': r'$\log_{10}(V_\mathrm{init})$'
    },
    'Vavg': {
        'latex': r'$V_\mathrm{avg}$',
        'scaled_latex': r'$\log_{10}(V_\mathrm{avg})$'
    },
    'TrailLength': {
        'latex': r'$L_\mathrm{trail}$',
        'scaled_latex': r'$\log_{10}(L_\mathrm{trail})$'
    },
    'Duration': {
        'latex': r'$t_\mathrm{dur}$',
        'scaled_latex': r'$t_\mathrm{dur}$'
    },
    'ZenithAngle': {
        'latex': r'$\theta_z$',
        'scaled_latex': r'$\cos(\theta_z)$'
    },
    'AbsMag': {
        'latex': r'$M_\mathrm{abs}$',
        'scaled_latex': r'$M_\mathrm{abs}$'
    },
    'F': {
        'latex': r'$F$',
        'scaled_latex': r'$F$'
    },
    'Mass_kg': {
        'latex': r'$m$',
        'scaled_latex': r'$\log_{10}(m)$'
    },
    'Deceleration': {
        'latex': r'$a_\mathrm{decel}$',
        'scaled_latex': r'$\log_{10}(a_\mathrm{decel})$'
    }
}

# Extract lists if needed
features = list(features_info.keys())
latex_symbols = {k: v['latex'] for k, v in features_info.items()}
scaled_latex_symbols = {k: v['scaled_latex'] for k, v in features_info.items()}


@dataclass(frozen=True)

class GMMParams:
    gmm_weights: np.ndarray
    gmm_means: np.ndarray
    gmm_covariances: np.ndarray


GMM_CONFIGS: Dict[int, GMMParams] = {
    3: GMMParams(
        gmm_weights=np.array([0.41796162,
                              0.34278431,
                              0.23925408
                              ]),

        gmm_means=np.array([[-0.83022888, -0.28674298, -0.11302983],
                            [0.70900848, -0.53354471, 0.32898939],
                            [0.43454568, 1.26534235, -0.27389407]
                            ]),

        gmm_covariances=np.array([[[2.81734610e-02, -4.88653369e-04, -1.12962089e-02],
                                   [-4.88653369e-04, 2.55463997e-01, 2.88399320e-02],
                                   [-1.12962089e-02, 2.88399320e-02, 7.41110478e-01]],

                                  [[7.65218051e-01, -1.40234363e-01, -3.98328455e-02],
                                   [-1.40234363e-01, 5.92287968e-01, 1.83163505e-01],
                                   [-3.98328455e-02, 1.83163505e-01, 1.27065171e+00]],

                                  [[9.20625448e-01, -2.21439813e-01, -3.02287867e-01],
                                   [-2.21439813e-01, 7.26172170e-01, 2.25476915e-01],
                                   [-3.02287867e-01, 2.25476915e-01, 8.05598015e-01]],
                                  ])
    ),

    11: GMMParams(
        gmm_weights=np.array([0.19519483, 0.03632081, 0.10158541, 0.0377207,
                              0.1930034, 0.09585493, 0.05098616, 0.08434407,
                              0.05962311, 0.09534943, 0.05001716
                              ]),

        gmm_means=np.array([[-0.85136621, -0.24148004, -0.70421087],
                            [0.04254277, 0.97892327, 1.41754598],
                            [0.00486575, -0.56905512, -0.51093435],
                            [1.26942937, 1.22519887, -0.94574602],
                            [-0.82664826, -0.24246533, 0.30471965],
                            [0.90835717, -0.71277956, 0.63000584],
                            [-0.76606149, -0.54692038, 1.52777391],
                            [-0.19829437, 1.80178582, -0.29261296],
                            [1.63328065, -1.21364997, -0.0196204],
                            [0.61781199, 0.44382246, -0.18909387],
                            [1.76396772, 0.88472734, 0.40652344]
                            ]),

        gmm_covariances=np.array([[[2.46707022e-02, -1.23853834e-03, -1.96850265e-02],
                                   [-1.23853834e-03, 2.47036363e-01, 7.39676488e-02],
                                   [-1.96850265e-02, 7.39676488e-02, 1.90197948e-01]],

                                  [[4.54888467e-01, -2.68215209e-01, 1.09873131e-01],
                                   [-2.68215209e-01, 1.70876411e+00, -3.71064363e-01],
                                   [1.09873131e-01, -3.71064363e-01, 1.10899452e+00]],

                                  [[2.00612222e-01, -2.27309910e-02, 2.24217367e-02],
                                   [-2.27309910e-02, 2.26113740e-01, 8.06802450e-02],
                                   [2.24217367e-02, 8.06802450e-02, 5.31648330e-01]],

                                  [[7.88655009e-01, -3.88182591e-01, -7.36928687e-01],
                                   [-3.88182591e-01, 5.44790577e-01, 5.20428194e-01],
                                   [-7.36928687e-01, 5.20428194e-01, 1.22536367e+00]],

                                  [[2.77301876e-02, 6.68284232e-03, -1.22476665e-02],
                                   [6.68284232e-03, 2.58858876e-01, 2.36368558e-02],
                                   [-1.22476665e-02, 2.36368558e-02, 4.27155224e-01]],

                                  [[3.44744611e-01, -1.28457018e-01, -2.68145784e-02],
                                   [-1.28457018e-01, 2.96660472e-01, 2.93133697e-02],
                                   [-2.68145784e-02, 2.93133697e-02, 5.88113076e-01]],

                                  [[3.61636739e-02, 3.21434143e-03, 1.04219699e-03],
                                   [3.21434143e-03, 6.97404530e-01, 2.42853395e-01],
                                   [1.04219699e-03, 2.42853395e-01, 1.19989329e+00]],

                                  [[4.09163595e-01, -2.09358089e-02, -1.93629015e-01],
                                   [-2.09358089e-02, 2.78398159e-01, 4.95791222e-02],
                                   [-1.93629015e-01, 4.95791222e-02, 3.69724945e-01]],

                                  [[3.97853037e-01, -1.47777644e-01, -2.28815777e-03],
                                   [-1.47777644e-01, 2.89511515e-01, 1.21113564e-01],
                                   [-2.28815777e-03, 1.21113564e-01, 8.77574771e-01]],

                                  [[4.01795791e-02, -5.80259811e-02, -2.74640646e-02],
                                   [-5.80259811e-02, 2.86310476e-01, 5.29328318e-02],
                                   [-2.74640646e-02, 5.29328318e-02, 6.30005184e-01]],

                                  [[5.70409728e-01, 2.78184940e-03, 2.77196451e-02],
                                   [2.78184940e-03, 6.43335648e-01, -4.40125717e-02],
                                   [2.77196451e-02, -4.40125717e-02, 1.06573178e+00]],
                                  ]),
    )
}


class HClassModel(object):

    def __init__(self, n_clusters: int):
        super().__setattr__("_frozen", False)

        if n_clusters not in GMM_CONFIGS:
            raise ValueError(
                f"n_clusters must be one of {tuple(GMM_CONFIGS.keys())}, got {n_clusters!r}"
            )

        self.features = features
        self.latex_symbols = latex_symbols
        self.scaled_latex_symbols = scaled_latex_symbols
        self.cluster_labels = {k: f"{n_clusters}-{string.ascii_uppercase[k]}"
                               for k in range(n_clusters)
                               }
        self.normalization_min = np.array([-0.80400014, -8.37307706, 1.77013815, 1.67034514,
                                           1.04601261, 1.01178521, 0.38372814, 0.2,
                                           0.04582851, -9.39, 0., -5.80410035, -4.61278386
                                           ])

        self.normalization_max = np.array([2.84297654, -3.48330594, 2.13747786, 2.0570831,
                                           1.88666565, 1.86380502, 2.31833898, 4.8,
                                           0.99998911, 2.83, 1., -0.79588002, 1.67488333
                                           ])

        self.fa_means = np.array([-0.28092403, -0.25107786, 0.32499514, 0.47153212,
                                  0.4826266, 0.51772416, -0.07208113, -0.8742448,
                                  0.36454656, 0.48804189, 0.171796, -0.37565354,
                                  0.56331339
                                  ])

        self.fa_components = np.array([[0.08455906, 0.23780462, -0.17516092, -0.18134596,
                                        -0.4046689, -0.40018603, -0.05406439,  0.069724,
                                        0.0687579, 0.08187245, -0.00980684, 0.0987227,
                                        -0.08597392],

                                       [0.16540866, 0.13742304, -0.10347116, -0.06159221,
                                        -0.0049407, -0.00860136, -0.06148199, -0.03218973,
                                        -0.0210266, -0.00567357, -0.06964115, -0.01101272,
                                        0.03648469],

                                       [0.01064679, -0.03751511,  0.02902104, -0.03750781,
                                        0.00646431,  0.01195469, 0.19556488,  0.10761494,
                                        -0.242326, -0.06364635, 0.05369203, 0.12692187,
                                        -0.09204429]
                                       ])

        self.fa_noise = np.array([5.66713919e-03, 6.31336263e-04, 1.01847910e-05, 1.00689688e-02,
                                  1.16785196e-05, 2.90494937e-04, 4.67357712e-05, 2.69195025e-03,
                                  1.30714148e-01, 1.95302567e-02, 1.10955420e-01, 1.82986733e-02,
                                  1.42251267e-02
                                  ])

        gmm = GMM_CONFIGS[n_clusters]
        self.gmm_weights = gmm.gmm_weights
        self.gmm_means = gmm.gmm_means
        self.gmm_covariances = gmm.gmm_covariances

        super().__setattr__("_frozen", True)

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError(f"Cannot modify attribute '{name}' — model is immutable.")
        super().__setattr__(name, value)
