from contextlib import nullcontext
from types import SimpleNamespace

from grasp_vision_pkg.camera_subscriber import CameraSubscriber
from grasp_vision_pkg.grasp_pose_estimator import GraspPose
import numpy as np
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header


class _Logger:

    def info(self, _message):
        pass

    def warn(self, _message):
        pass

    def error(self, _message):
        pass


class _Response:
    STATUS_SUCCESS = 0
    STATUS_NOT_READY = 1
    STATUS_INVALID_REQUEST = 2
    STATUS_ESTIMATOR_UNAVAILABLE = 3
    STATUS_ESTIMATION_FAILED = 4
    STATUS_INTERNAL_ERROR = 5

    def __init__(self):
        self.success = False
        self.status_code = 0
        self.message = ''
        self.pose = None
        self.object_pose = None
        self.object_dimensions = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        self.object_cloud = PointCloud2()
        self.background_cloud = PointCloud2()
        self.has_segmented_clouds = False
        self.width = 0.0
        self.score = 0.0
        self.point_count = 0
        self.mask_pixel_count = 0
        self.segmentation_score = 0.0
        self.debug_image_path = ''
        self.processing_time = SimpleNamespace(sec=0, nanosec=0)


class _Estimator:

    def __init__(self, pose):
        self.pose = pose
        self.min_depth = 0.05
        self.max_depth = 3.0
        self.last_mask = np.array([[True, False], [True, True]])
        self.last_points = np.array([
            [0.08, 0.18, 0.28],
            [0.12, 0.22, 0.32],
            [0.10, 0.20, 0.30],
        ])
        self.segmenter = SimpleNamespace(
            last_prediction=SimpleNamespace(
                scores=np.array([0.12, 0.87]),
                selected_index=1,
            )
        )

    def estimate(self, **_kwargs):
        return self.pose

    def _normalize_camera_matrix(self, matrix):
        return np.asarray(matrix, dtype=np.float64)

    def _depth_to_meters(self, depth):
        return np.asarray(depth, dtype=np.float32)


def _request(**overrides):
    values = {
        'use_default_prompt': True,
        'prompt_type': 'text',
        'prompt': 'visual',
        'box_prompt': [0.0, 0.0, 0.0, 0.0],
        'publish_result': False,
        'save_debug_image': False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _node_without_ros():
    node = CameraSubscriber.__new__(CameraSubscriber)
    node.latest_color_frame = None
    node.latest_aligned_depth_frame = None
    node.color_camera_matrix = None
    node.aligned_depth_camera_matrix = None
    node.latest_color_header = None
    node.latest_aligned_depth_header = None
    node.default_grasp_prompt = 'visual'
    node.sam3_segmenter = None
    node.grasp_estimator = None
    node.grasp_estimator_error = ''
    node.grasp_estimator_lock = nullcontext()
    node.publish_grasp_result = False
    node.save_grasp_debug_image = False
    node.segmented_cloud_stride = 1
    node.background_mask_dilation_px = 0
    node.background_object_bbox_filter = False
    node.background_object_bbox_padding_xy = 0.0
    node.background_object_bbox_padding_z = 0.0
    node.min_object_dimension = 0.01
    node.grasp_frame_id = ''
    node.get_logger = lambda: _Logger()
    return node


def test_estimate_grasp_pose_reports_not_ready():
    node = _node_without_ros()

    response = node._estimate_grasp_pose(_request(), _Response())

    assert not response.success
    assert response.status_code == response.STATUS_NOT_READY
    assert 'Waiting for color image' in response.message


def test_estimate_grasp_pose_rejects_invalid_prompt():
    node = _node_without_ros()
    node.latest_color_frame = np.zeros((2, 2, 3), dtype=np.uint8)
    node.latest_aligned_depth_frame = np.ones((2, 2), dtype=np.uint16)
    node.color_camera_matrix = np.eye(3)

    response = node._estimate_grasp_pose(
        _request(use_default_prompt=False, prompt_type='point'),
        _Response(),
    )

    assert not response.success
    assert response.status_code == response.STATUS_INVALID_REQUEST
    assert 'Invalid grasp prompt' in response.message


def test_estimate_grasp_pose_success_response_contains_status_details():
    pose = GraspPose(
        position=np.array([0.1, 0.2, 0.3]),
        orientation_matrix=np.eye(3),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        width=0.06,
        score=0.91,
        center_pixel=(1, 1),
        point_count=42,
    )
    node = _node_without_ros()
    node.latest_color_frame = np.zeros((2, 2, 3), dtype=np.uint8)
    node.latest_aligned_depth_frame = np.ones((2, 2), dtype=np.uint16)
    node.color_camera_matrix = np.eye(3)
    node.latest_color_header = Header()
    node.latest_color_header.frame_id = 'camera_color'
    node.grasp_estimator = _Estimator(pose)
    node._ensure_grasp_estimator = lambda: True

    response = node._estimate_grasp_pose(_request(), _Response())

    assert response.success
    assert response.status_code == response.STATUS_SUCCESS
    assert response.pose.header.frame_id == 'camera_color'
    assert response.pose.pose.position.z == 0.3
    assert np.isclose(response.object_pose.pose.position.z, 0.3)
    assert np.isclose(response.object_dimensions.x, 0.04)
    assert response.width == 0.06
    assert response.score == 0.91
    assert response.point_count == 42
    assert response.mask_pixel_count == 3
    assert response.segmentation_score == 0.87
    assert response.has_segmented_clouds
    assert response.object_cloud.header.frame_id == 'camera_color'
    assert response.object_cloud.width == 3
    assert response.background_cloud.header.frame_id == 'camera_color'
    assert response.background_cloud.width == 1


def test_estimate_grasp_pose_releases_cached_inference_outputs():
    pose = GraspPose(
        position=np.array([0.1, 0.2, 0.3]),
        orientation_matrix=np.eye(3),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        width=0.06,
        score=0.91,
        center_pixel=(1, 1),
        point_count=42,
    )
    node = _node_without_ros()
    node.latest_color_frame = np.zeros((2, 2, 3), dtype=np.uint8)
    node.latest_aligned_depth_frame = np.ones((2, 2), dtype=np.uint16)
    node.color_camera_matrix = np.eye(3)
    node.latest_color_header = Header()
    node.latest_color_header.frame_id = 'camera_color'
    node.grasp_estimator = _Estimator(pose)
    node._ensure_grasp_estimator = lambda: True

    response = node._estimate_grasp_pose(_request(), _Response())

    assert response.success
    assert node.grasp_estimator.last_mask is None
    assert node.grasp_estimator.last_points is None
    assert node.grasp_estimator.segmenter.last_prediction is None


def test_ensure_sam3_segmenter_reuses_cached_instance():
    node = _node_without_ros()
    created = []

    def create_segmenter():
        segmenter = SimpleNamespace(name=f'segmenter-{len(created)}')
        created.append(segmenter)
        return segmenter

    node._create_sam3_segmenter = create_segmenter

    first = node._ensure_sam3_segmenter()
    second = node._ensure_sam3_segmenter()

    assert first is second
    assert created == [first]


def test_ensure_grasp_estimator_reuses_cached_instance():
    node = _node_without_ros()
    created = []

    def create_estimator():
        estimator = SimpleNamespace(name=f'estimator-{len(created)}')
        created.append(estimator)
        return estimator

    node._create_grasp_estimator = create_estimator

    assert node._ensure_grasp_estimator()
    first = node.grasp_estimator
    assert node._ensure_grasp_estimator()

    assert node.grasp_estimator is first
    assert created == [first]


def test_background_cloud_removes_padded_object_bbox_volume():
    node = _node_without_ros()
    node.background_object_bbox_filter = True
    node.background_object_bbox_padding_xy = 0.2
    node.background_object_bbox_padding_z = 0.2
    node.grasp_estimator = SimpleNamespace(
        min_depth=0.05,
        max_depth=3.0,
        _normalize_camera_matrix=lambda matrix: np.asarray(matrix, dtype=np.float64),
        _depth_to_meters=lambda depth: np.asarray(depth, dtype=np.float32),
    )

    depth = np.ones((3, 3), dtype=np.float32)
    color = np.zeros((3, 3, 3), dtype=np.uint8)
    intrinsics = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
    ])
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True

    object_cloud, background_cloud = node._make_segmented_clouds(
        color,
        depth,
        intrinsics,
        mask,
        Header(),
    )

    assert object_cloud.width == 1
    assert background_cloud.width == 8

    node.background_object_bbox_padding_xy = 1.1
    object_cloud, background_cloud = node._make_segmented_clouds(
        color,
        depth,
        intrinsics,
        mask,
        Header(),
    )

    assert object_cloud.width == 1
    assert background_cloud.width == 0
