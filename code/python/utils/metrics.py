import numpy as np

def compute_rmse(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute the Root Mean Square Error (RMSE) between two point clouds A and B.
    Both must be of shape Nx2.
    """
    if A.shape != B.shape:
        raise ValueError("Point clouds must have the same shape.")
    diff = A - B
    return np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))

def mean_distance(points: np.ndarray) -> float:
    """
    Compute the average pairwise distance between points.
    """
    from scipy.spatial.distance import pdist
    return np.mean(pdist(points))
