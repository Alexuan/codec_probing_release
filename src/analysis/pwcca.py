# Vendored from vocal_tract_distance/lib/Projection_Weighted_CCA/cca.py,
# which is itself adapted from https://github.com/google/svcca (Apache 2.0).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import numpy as np


def positivedef_matrix_sqrt(array):
    w, v = np.linalg.eigh(array)
    wsqrt = np.sqrt(w)
    return np.dot(v, np.dot(np.diag(wsqrt), np.conj(v).T))


def remove_small(sigma_xx, sigma_xy, sigma_yx, sigma_yy, threshold=1e-6):
    x_diag = np.abs(np.diagonal(sigma_xx))
    y_diag = np.abs(np.diagonal(sigma_yy))
    x_idxs = x_diag >= threshold
    y_idxs = y_diag >= threshold
    return (
        sigma_xx[x_idxs][:, x_idxs],
        sigma_xy[x_idxs][:, y_idxs],
        sigma_yx[y_idxs][:, x_idxs],
        sigma_yy[y_idxs][:, y_idxs],
        x_idxs,
        y_idxs,
    )


def gs_orthonormalize(array):
    q, _ = np.linalg.qr(array)
    if q.shape[1] < array.shape[1]:
        zero_pad = np.zeros(shape=(q.shape[0], array.shape[1] - q.shape[1]))
        q = np.concatenate([q, zero_pad], 1)
    return q


def solve_cca(x: np.ndarray, y: np.ndarray) -> dict:
    """CCA correlations, position vectors, images, and PWCCA / mean similarity.

    Args:
        x: (num_neurons, num_datapoints)
        y: (num_neurons, num_datapoints)
    """
    assert x.ndim == y.ndim == 2
    assert x.shape[1] == y.shape[1]
    assert x.shape[0] <= x.shape[1]
    assert y.shape[0] <= y.shape[1]
    epsilon = 1e-6

    numx = x.shape[0]

    sigma = np.cov(x, y)
    sigmaxx = sigma[:numx, :numx]
    sigmaxy = sigma[:numx, numx:]
    sigmayx = sigma[numx:, :numx]
    sigmayy = sigma[numx:, numx:]

    xmax, ymax = np.max(np.abs(sigmaxx)), np.max(np.abs(sigmayy))
    sigmaxx /= xmax
    sigmayy /= ymax
    sigmaxy /= np.sqrt(xmax * ymax)
    sigmayx /= np.sqrt(ymax * xmax)

    sigmaxx, sigmaxy, sigmayx, sigmayy, x_idxs, y_idxs = remove_small(
        sigmaxx, sigmaxy, sigmayx, sigmayy
    )
    x = x[x_idxs]
    y = y[y_idxs]

    numx = sigmaxx.shape[0]
    numy = sigmayy.shape[0]
    if numx == 0 or numy == 0:
        raise ValueError("CCA: covariance matrices are empty after small-value pruning")

    sigmaxx += epsilon * np.eye(numx)
    sigmayy += epsilon * np.eye(numy)
    inv_sigmaxx = np.linalg.pinv(sigmaxx)
    inv_sigmayy = np.linalg.pinv(sigmayy)
    invsqrt_sigmaxx = positivedef_matrix_sqrt(inv_sigmaxx)
    invsqrt_sigmayy = positivedef_matrix_sqrt(inv_sigmayy)

    arrx = invsqrt_sigmaxx.dot(sigmaxy).dot(inv_sigmayy.dot(sigmayx.dot(invsqrt_sigmaxx)))
    arry = invsqrt_sigmayy.dot(sigmayx).dot(inv_sigmaxx.dot(sigmaxy.dot(invsqrt_sigmayy)))
    arrx += epsilon * np.eye(arrx.shape[0])
    arry += epsilon * np.eye(arry.shape[0])

    _, sx, vhx = np.linalg.svd(arrx)
    _, sy, vhy = np.linalg.svd(arry)

    def _clean(s):
        s = np.sqrt(np.abs(s))
        s = np.where(s > 1, 1, s)
        return np.where(s < epsilon, 0, s)

    cca_corr_x = _clean(sx)
    cca_corr_y = _clean(sy)

    cca_pos_x = vhx.dot(invsqrt_sigmaxx)
    cca_pos_y = vhy.dot(invsqrt_sigmayy)
    cca_image_x = cca_pos_x.dot(x)
    cca_image_y = cca_pos_y.dot(y)

    min_numxy = min(numx, numy)
    truncated_corr_x = cca_corr_x[:min_numxy]
    truncated_corr_y = cca_corr_y[:min_numxy]

    orthonorm_x = gs_orthonormalize(cca_image_x[:min_numxy])
    orthonorm_y = gs_orthonormalize(cca_image_y[:min_numxy])

    wx = np.abs(orthonorm_x.dot(x.T)).sum(1)
    wx /= wx.sum()
    wy = np.abs(orthonorm_y.dot(y.T)).sum(1)
    wy /= wy.sum()

    return {
        "cca_corr_x": cca_corr_x,
        "cca_corr_y": cca_corr_y,
        "cca_pos_x": cca_pos_x,
        "cca_pos_y": cca_pos_y,
        "cca_image_x": cca_image_x,
        "cca_image_y": cca_image_y,
        "ewcca_sim_x": truncated_corr_x.mean(),
        "ewcca_sim_y": truncated_corr_y.mean(),
        "pwcca_sim_x": (wx * truncated_corr_x).sum(),
        "pwcca_sim_y": (wy * truncated_corr_y).sum(),
    }
