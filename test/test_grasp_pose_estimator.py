import math

from grasp_vision_pkg.grasp_pose_estimator import GraspPoseEstimator
import numpy as np
import pytest


def _make_test_data():
    height = 120
    width = 160
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :] = (20, 30, 40)

    depth = np.full((height, width), 1000, dtype=np.uint16)
    yy, xx = np.ogrid[:height, :width]
    mask = ((xx - 80) ** 2) / (28.0 ** 2) + ((yy - 60) ** 2) / (14.0 ** 2) <= 1.0
    mask = mask.astype(np.uint8) * 255
    depth[mask.astype(bool)] = 800

    camera_matrix = np.array(
        [[200.0, 0.0, 80.0], [0.0, 200.0, 60.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rgb, depth, mask, camera_matrix


def test_estimate_from_precomputed_mask():
    rgb, depth, mask, camera_matrix = _make_test_data()
    estimator = GraspPoseEstimator(min_points=20)

    pose = estimator.estimate(
        rgb,
        depth,
        camera_matrix,
        object_mask=mask,
    )

    assert pose.point_count >= 20
    assert np.isfinite(pose.position).all()
    assert np.isfinite(pose.quaternion_xyzw).all()
    assert math.isclose(np.linalg.norm(pose.quaternion_xyzw), 1.0, rel_tol=1e-6)
    assert pose.width > 0.0
    assert 0.0 <= pose.score <= 1.0
    assert pose.center_pixel[0] == 80
    assert pose.center_pixel[1] == 60

    vis = estimator.visualize(rgb, pose, camera_matrix, object_mask=mask)
    assert vis.shape == rgb.shape
    assert vis.dtype == np.uint8
    assert not np.array_equal(vis, rgb)


def test_estimate_rejects_mask_shape_mismatch():
    rgb, depth, mask, camera_matrix = _make_test_data()
    estimator = GraspPoseEstimator(min_points=20)

    with pytest.raises(ValueError, match='does not match image shape'):
        estimator.estimate(rgb, depth, camera_matrix, object_mask=mask[:100])


def test_estimate_rejects_too_few_points():
    rgb, depth, mask, camera_matrix = _make_test_data()
    estimator = GraspPoseEstimator(min_points=5000)

    with pytest.raises(ValueError, match='Not enough valid object depth points'):
        estimator.estimate(rgb, depth, camera_matrix, object_mask=mask)


def test_estimate_rejects_empty_depth():
    rgb, depth, mask, camera_matrix = _make_test_data()
    depth = np.zeros_like(depth)
    estimator = GraspPoseEstimator(min_points=20)

    with pytest.raises(ValueError, match='Not enough valid object depth points'):
        estimator.estimate(rgb, depth, camera_matrix, object_mask=mask)
