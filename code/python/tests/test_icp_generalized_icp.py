import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from tests.common import make_wireframe_cube, apply_known_transform, visualize_icp_result
from icp.config import ICPConfig
from icp.generalized_icp import icp

def test_icp_alignment():
    map_ref = make_wireframe_cube(center=(0, 0, 0), size=2.0)
    scan = apply_known_transform(map_ref, dx=0.3, dy=0.2, dz=0.1, rz=np.pi / 18)

    config = ICPConfig()
    aligned, updated = icp(scan, map_ref, config=config)

    visualize_icp_result(map_ref, scan, aligned)

    assert aligned.shape == scan.shape
    assert updated.shape[0] >= aligned.shape[0]
