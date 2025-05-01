import numpy as np

def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    Apply a 3x3 or 4x4 homogeneous transformation matrix to a set of 2D or 3D points.

    Args:
        points: (N, 2) or (N, 3) array of points.
        transform: 3x3 (for 2D) or 4x4 (for 3D) homogeneous transformation matrix.

    Returns:
        Transformed points as an (N, 2) or (N, 3) array.
    """
    if points.ndim != 2 or transform.ndim != 2:
        raise ValueError("Invalid input dimensions")

    n_dims = points.shape[1]
    if n_dims == 2 and transform.shape != (3, 3):
        raise ValueError("2D transform must be 3x3")
    if n_dims == 3 and transform.shape != (4, 4):
        raise ValueError("3D transform must be 4x4")

    homog = np.hstack([points, np.ones((len(points), 1))])  # (N, 3) or (N, 4)
    transformed = homog @ transform.T
    return transformed[:, :n_dims]

def build_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Build a homogeneous transformation matrix from rotation and translation.

    Args:
        R: Rotation matrix (2x2 or 3x3)
        t: Translation vector (length 2 or 3)

    Returns:
        3x3 or 4x4 homogeneous transformation matrix
    """
    if R.shape == (2, 2) and t.shape == (2,):
        T = np.eye(3)
        T[:2, :2] = R
        T[:2, 2] = t
        return T
    elif R.shape == (3, 3) and t.shape == (3,):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    else:
        raise ValueError("Invalid shape for rotation or translation")
