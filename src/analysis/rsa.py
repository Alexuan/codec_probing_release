"""Representational Similarity Analysis: correlation-distance RDMs compared by Spearman."""

import numpy as np


def rsa_similarity(X: np.ndarray, Y: np.ndarray) -> float:
    """RSA score for inputs shaped ``(num_samples, num_features)``."""
    import rsatoolbox

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of samples for RSA.")

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

    ds_x = rsatoolbox.data.Dataset(X)
    ds_y = rsatoolbox.data.Dataset(Y)

    rdm_x = rsatoolbox.rdm.calc_rdm(ds_x, method="correlation")
    rdm_y = rsatoolbox.rdm.calc_rdm(ds_y, method="correlation")
    score = rsatoolbox.rdm.compare(rdm_x, rdm_y, method="spearman")
    return float(np.asarray(score).squeeze())
