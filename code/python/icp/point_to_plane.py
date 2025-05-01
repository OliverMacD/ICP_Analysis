import numpy as np
from icp.icp import ICPAlgorithm
from icp.config import ICPConfig
from utils.transform import apply_transform, build_transform
from sklearn.neighbors import NearestNeighbors


class PointToPlaneICP(ICPAlgorithm):
    def __init__(self, config: ICPConfig):
        super().__init__(config)

    def _compute_normals(self, points: np.ndarray, k: int = 10) -> np.ndarray:
        dim = points.shape[1]
        if dim == 2:
            # Estimate 2D normals (tangent perpendicular)
            nbrs = NearestNeighbors(n_neighbors=k).fit(points)
            _, indices = nbrs.kneighbors(points)

            normals = []
            for idxs in indices:
                local = points[idxs]
                cov = np.cov(local.T)
                eigvals, eigvecs = np.linalg.eigh(cov)
                normal = eigvecs[:, 0]
                normal /= np.linalg.norm(normal)
                normals.append(normal)
            return np.array(normals)

        elif dim == 3:
            # Estimate 3D normals via PCA
            nbrs = NearestNeighbors(n_neighbors=k).fit(points)
            _, indices = nbrs.kneighbors(points)

            normals = []
            for idxs in indices:
                local = points[idxs]
                cov = np.cov(local.T)
                eigvals, eigvecs = np.linalg.eigh(cov)
                normal = eigvecs[:, 0]  # Smallest eigenvector
                normal /= np.linalg.norm(normal)
                normals.append(normal)
            return np.array(normals)

        else:
            raise ValueError("Only 2D and 3D points supported for normal estimation.")

    def _skew(self, v: np.ndarray) -> np.ndarray:
        x, y, z = v.flatten()
        return np.array([
            [0, -z,  y],
            [z,  0, -x],
            [-y, x,  0]
        ])

    def register(self, new_scan: np.ndarray, map_so_far: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        filtered_scan = self._apply_gaussian_filter(new_scan)
        reference = self._get_reference_map(map_so_far)

        # Match points
        src_pts, tgt_pts = self._find_neighbors(filtered_scan, reference)

        if len(src_pts) < 3:
            print("Not enough matches.")
            return filtered_scan, map_so_far

        # Reject pairs
        src_pts, tgt_pts = self._reject_pairs(src_pts, tgt_pts)

        if len(src_pts) < 3:
            print("Not enough valid pairs after rejection.")
            return filtered_scan, map_so_far

        # Compute normals of target (map) points
        normals = self._compute_normals(tgt_pts)

        dim = src_pts.shape[1]
        A = []
        b = []

        for i in range(len(src_pts)):
            pi = src_pts[i]
            qi = tgt_pts[i]
            ni = normals[i]

            if dim == 2:
                px, py = pi
                nx, ny = ni

                # Jacobian: [∂e/∂θ, ∂e/∂tx, ∂e/∂ty]
                dtheta = (-py * nx + px * ny)
                A.append([dtheta, nx, ny])
                b.append(np.dot(ni, qi - pi))

            elif dim == 3:
                skew_pi = self._skew(pi)  # 3x3
                J_rot = -ni @ skew_pi     # (1x3)
                J_trans = ni              # (1x3)
                A.append(np.hstack([J_rot, J_trans]))
                b.append(np.dot(ni, qi - pi))

            else:
                raise ValueError("Only 2D and 3D supported.")

        A = np.vstack(A)
        b = np.array(b).reshape(-1, 1)

        try:
            delta, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            print("Point-to-Plane ICP: Singular matrix.")
            return filtered_scan, map_so_far

        if dim == 2:
            dtheta, dx, dy = delta.flatten()
            cos_t, sin_t = np.cos(dtheta), np.sin(dtheta)
            R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            t = np.array([dx, dy])

        elif dim == 3:
            w = delta[:3].flatten()
            t = delta[3:].flatten()
            angle = np.linalg.norm(w)

            if angle < 1e-6:
                R = np.eye(3)
            else:
                axis = w / angle
                x, y, z = axis
                K = np.array([
                    [0, -z,  y],
                    [z,  0, -x],
                    [-y, x,  0]
                ])
                R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

        else:
            raise ValueError("Unsupported dimension")

        T = build_transform(R, t)
        aligned = apply_transform(filtered_scan, T)
        updated_map = np.vstack([map_so_far, aligned])
        self.prev_map = reference

        return aligned, updated_map


def icp(new_scan: np.ndarray, map_so_far: np.ndarray, config: ICPConfig = ICPConfig()):
    return PointToPlaneICP(config).register(new_scan, map_so_far)
