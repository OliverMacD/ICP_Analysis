# Notes

## To run tests
pytest -s tests/test_icp_generalized_icp.py
pytest -s tests/test_icp_point_to_point.py
pytest -s tests/test_icp_point_to_plane.py
pytest -s tests/

# Initial Tests

## tests\test_icp_point_to_plane.py
====== 2D Point-to-Plane ICP Test ======
[2D Config]
ICPConfig(segmentation_strategy='none', gaussian_filter_pct=0.0, memory_mode='full', rejection={'max_normal_diff': None, 'max_distance': None, 'percentile_clip': None}, neighborhood_mode='fixed', search_radius=0.5, density_factor=1.0, num_segments=4)
[2D RMSE Error] 0.008285
.
====== 3D Point-to-Plane ICP Test ======
[3D Config]
ICPConfig(segmentation_strategy='none', gaussian_filter_pct=0.0, memory_mode='full', rejection={'max_normal_diff': None, 'max_distance': None, 'percentile_clip': None}, neighborhood_mode='fixed', search_radius=0.5, density_factor=1.0, num_segments=4)
[3D RMSE Error] 0.224741

## tests\test_icp_point_to_point.py
====== 2D Point-to-Point ICP Test ======
[2D Config]
ICPConfig(segmentation_strategy='none', gaussian_filter_pct=0.0, memory_mode='full', rejection={'max_normal_diff': None, 'max_distance': None, 'percentile_clip': None}, neighborhood_mode='fixed', search_radius=0.5, density_factor=1.0, num_segments=4)
[2D RMSE Error] 0.003000
.
====== 3D Point-to-Point ICP Test ======
[3D Config]
ICPConfig(segmentation_strategy='none', gaussian_filter_pct=0.0, memory_mode='full', rejection={'max_normal_diff': None, 'max_distance': None, 'percentile_clip': None}, neighborhood_mode='fixed', search_radius=0.5, density_factor=1.0, num_segments=4)
[3D RMSE Error] 0.254130

## tests\test_icp_generalized_icp.py
====== 2D Generalized ICP Test ======
[2D Config]
ICPConfig(segmentation_strategy='none', gaussian_filter_pct=0.0, memory_mode='full', rejection={'max_normal_diff': None, 'max_distance': None, 'percentile_clip': None}, neighborhood_mode='fixed', search_radius=0.5, density_factor=1.0, num_segments=4)
[2D RMSE Error] 0.008210
.
====== 3D Generalized ICP Test ======
[3D Config]
ICPConfig(segmentation_strategy='none', gaussian_filter_pct=0.0, memory_mode='full', rejection={'max_normal_diff': None, 'max_distance': None, 'percentile_clip': None}, neighborhood_mode='fixed', search_radius=0.5, density_factor=1.0, num_segments=4)
[3D RMSE Error] 0.303808