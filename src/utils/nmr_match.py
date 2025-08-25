# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Optional


def set_match_score_weighted(x: np.ndarray, y: np.ndarray, weight_matrix: Optional[np.ndarray] = None, sigma: float = 1) -> float:
    """
    Computes a weighted matching score between two sets of NMR data.
    """
    if len(x) == 0 or len(y) == 0:
        return 1.0 if (len(x) == 0 and len(y) == 0) else 0.0
    
    dist_matrix = np.abs(x[:, np.newaxis] - y[np.newaxis, :])
    score_matrix = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
    if weight_matrix is not None:
        assert score_matrix.shape == weight_matrix.shape
        score_matrix *= weight_matrix
    
    cost_matrix = -score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_score = score_matrix[row_ind, col_ind].sum()
    
    score = total_score / np.sqrt(len(x) * len(y))
    return score


def set2vec(set_list: List, nmr_type: str, dim: int = 128, normalize: bool = True, sigma: Optional[float] = None) -> np.ndarray:
    """
    Converts a set of NMR data into a vector representation.
    """
    if isinstance(set_list, np.ndarray):
        return set2vec([set_list], nmr_type, dim, normalize)
    # set_list: list of np.array
    assert isinstance(set_list, list)
    if nmr_type == 'H':
        nmr_range = (-1, 15)
        sigma = sigma or 0.3
    elif nmr_type == 'C':
        nmr_range = (-10, 230)
        sigma = sigma or 2
    ni = np.linspace(nmr_range[0], nmr_range[1], dim)
    interval = ni[1] - ni[0]
    # gaussian kernel
    coef = interval / (np.sqrt(2 * np.pi) * sigma)
    
    result = [coef * np.exp(-(np.abs(item[:, np.newaxis] - ni) / sigma) ** 2 / 2).sum(axis=0) for item in set_list]
    
    result = np.array(result)
    if normalize:
        return normalize_vectors(result).astype(np.float32)
    else:
        return result.astype(np.float32)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalizes vectors.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    return vectors / norms
