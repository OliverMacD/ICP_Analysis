import numpy as np
from icp.icp import ICPAlgorithm
from icp.config import ICPConfig
from utils.transform import apply_transform, build_transform
from sklearn.neighbors import NearestNeighbors


class GeneralizedICP(ICPAlgorithm):
    def __init__(self, config: ICPConfig):
        super().__init__(config)

    def _compute_covariances(self, points: np.ndarray, k: int = 10) -> list[np.ndarray]:
        if len(points) < k:
            raise ValueError("Too few points for covariance estimation")

        nbrs = NearestNeighbors(n_neighbors=k).fit(points)
        _, indices = nbrs.kneighbors(points)

        covariances = []
        for idxs in indices:
            neighborhood = points[idxs]
            mean = np.mean(neighborhood, axis=0)
            centered = neighborhood - mean
            dim = centered.shape[1]
            cov = np.cov(centered.T) + 1e-6 * np.eye(dim)
            covariances.append(cov)

        return covariances

    def _skew(self, v: np.ndarray) -> np.ndarray:
        """
        Create a skew-symmetric matrix from a 3D vector for cross product.
        """
        x, y, z = v.flatten()
        return np.array([
            [ 0, -z,  y],
            [ z,  0, -x],
            [-y,  x,  0]
        ])

    def register(self, new_scan: np.ndarray, map_so_far: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        filtered_scan = self._apply_gaussian_filter(new_scan)
        reference = self._get_reference_map(map_so_far)

        src_pts, tgt_pts = self._find_neighbors(filtered_scan, reference)

        if len(src_pts) < 3:
            print("Not enough matches.")
            return filtered_scan, map_so_far

        src_pts, tgt_pts = self._reject_pairs(src_pts, tgt_pts)

        if len(src_pts) < 3:
            print("Not enough valid pairs after rejection.")
            return filtered_scan, map_so_far

        cov_src = self._compute_covariances(src_pts)
        cov_tgt = self._compute_covariances(tgt_pts)

        dim = src_pts.shape[1]

        if dim == 2:
            H = np.zeros((3, 3))
            g = np.zeros((3, 1))
        elif dim == 3:
            H = np.zeros((6, 6))
            g = np.zeros((6, 1))
        else:
            raise ValueError("Unsupported dimensionality")

        for i in range(len(src_pts)):
            pi = src_pts[i].reshape(dim, 1)
            qi = tgt_pts[i].reshape(dim, 1)
            Ci = cov_src[i] + cov_tgt[i]
            Ci_inv = np.linalg.inv(Ci)
            r = qi - pi

            if dim == 2:
                Ji = np.array([
                    [-pi[1, 0], 1, 0],
                    [ pi[0, 0], 0, 1]
                ])  # (2x3)
            else:  # dim == 3
                skew = self._skew(pi)
                Ji = np.hstack([skew, np.eye(3)])  # (3x6)

            H += Ji.T @ Ci_inv @ Ji
            g += Ji.T @ Ci_inv @ r

        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            print("G-ICP: Singular system.")
            return filtered_scan, map_so_far

        if dim == 2:
            dtheta, dx, dy = delta.flatten()
            cos_t, sin_t = np.cos(dtheta), np.sin(dtheta)
            R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            t = np.array([dx, dy])
        else:
            w = delta[:3].flatten()
            t = delta[3:].flatten()

            angle = np.linalg.norm(w)
            if angle < 1e-6:
                R = np.eye(3)
            else:
                axis = w / angle
                x, y, z = axis
                K = np.array([
                    [ 0, -z,  y],
                    [ z,  0, -x],
                    [-y,  x,  0]
                ])
                R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        T = build_transform(R, t)
        aligned = apply_transform(filtered_scan, T)
        updated_map = np.vstack([map_so_far, aligned])
        self.prev_map = reference

        return aligned, updated_map


def icp(new_scan: np.ndarray, map_so_far: np.ndarray, config: ICPConfig = ICPConfig()):
    return GeneralizedICP(config).register(new_scan, map_so_far)
