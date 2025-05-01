import numpy as np
from icp.icp import ICPAlgorithm
from icp.config import ICPConfig
from utils.transform import apply_transform, build_transform

class PointToPointICP(ICPAlgorithm):
    def __init__(self, config: ICPConfig):
        super().__init__(config)

    def register(self, new_scan: np.ndarray, map_so_far: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run one step of Point-to-Point ICP using configured filters and rejection.
        Returns the aligned scan and the updated map.
        """
        # Step 1: Apply Gaussian filtering to incoming scan
        filtered_scan = self._apply_gaussian_filter(new_scan)

        # Step 2: Select which part of the map to use
        reference = self._get_reference_map(map_so_far)

        # Step 3: Find correspondences
        src_pts, tgt_pts = self._find_neighbors(filtered_scan, reference)

        if len(src_pts) < 3:
            print("Not enough matched pairs for alignment.")
            return filtered_scan, map_so_far

        # Step 4: Reject bad pairs based on distance, normals, etc.
        src_pts, tgt_pts = self._reject_pairs(src_pts, tgt_pts)

        if len(src_pts) < 3:
            print("Not enough valid pairs after rejection.")
            return filtered_scan, map_so_far

        # Step 5: Compute alignment using SVD
        centroid_src = np.mean(src_pts, axis=0)
        centroid_tgt = np.mean(tgt_pts, axis=0)
        A = src_pts - centroid_src
        B = tgt_pts - centroid_tgt

        H = A.T @ B
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            R[:, -1] *= -1  # Fix reflection if needed

        t = centroid_tgt - R @ centroid_src
        T = build_transform(R, t)

        # Step 6: Transform entire (filtered) scan
        aligned_scan = apply_transform(filtered_scan, T)

        # Step 7: Update map by appending aligned points
        updated_map = np.vstack([map_so_far, aligned_scan])

        # Step 8: Save memory state if applicable
        self.prev_map = reference

        return aligned_scan, updated_map

# Required entrypoint for dynamic loading
def icp(new_scan: np.ndarray, map_so_far: np.ndarray, config: ICPConfig = ICPConfig()):
    return PointToPointICP(config).register(new_scan, map_so_far)
