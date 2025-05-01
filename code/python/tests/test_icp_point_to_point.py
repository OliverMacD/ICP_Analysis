import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from tests.common import (
    make_square_cloud,
    make_wireframe_cube,
    apply_known_transform,
    visualize_icp_result,
)
from icp.config import ICPConfig
from icp.point_to_point import icp


def test_icp_point_to_point_2d():
    print("\n====== 2D Point-to-Point ICP Test ======")

    map_ref = make_square_cloud(center=(0, 0), size=2.0)
    scan = apply_known_transform(map_ref, dx=0.3, dy=0.2, theta=np.pi / 18)

    config = ICPConfig()
    aligned, updated = icp(scan, map_ref, config=config)

    error = np.mean(np.linalg.norm(aligned - scan[:len(aligned)], axis=1))

    print("[2D Config]")
    print(config)
    print(f"[2D RMSE Error] {error:.6f}")

    visualize_icp_result(map_ref, scan, aligned)

    assert aligned.shape == scan.shape
    assert updated.shape[0] >= aligned.shape[0]
    assert error < 0.5


def test_icp_point_to_point_3d():
    print("\n====== 3D Point-to-Point ICP Test ======")

    map_ref = make_wireframe_cube(center=(0, 0, 0), size=2.0)
    scan = apply_known_transform(map_ref, dx=0.3, dy=0.2, dz=0.1, rz=np.pi / 18)

    config = ICPConfig()
    aligned, updated = icp(scan, map_ref, config=config)

    error = np.mean(np.linalg.norm(aligned - scan[:len(aligned)], axis=1))

    print("[3D Config]")
    print(config)
    print(f"[3D RMSE Error] {error:.6f}")

    visualize_icp_result(map_ref, scan, aligned)

    assert aligned.shape == scan.shape
    assert updated.shape[0] >= aligned.shape[0]
    assert error < 0.5
