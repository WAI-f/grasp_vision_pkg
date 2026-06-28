from types import SimpleNamespace

from grasp_vision_pkg.camera_subscriber import CameraSubscriber
from grasp_vision_pkg.grasp_pose_estimator import GraspPose
import numpy as np
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
        self.last_mask = np.array([[True, False], [True, True]])
        self.segmenter = SimpleNamespace(
            last_prediction=SimpleNamespace(
                scores=np.array([0.12, 0.87]),
                selected_index=1,
            )
        )

    def estimate(self, **_kwargs):
        return self.pose


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
    node.grasp_estimator = None
    node.grasp_estimator_error = ''
    node.publish_grasp_result = False
    node.save_grasp_debug_image = False
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
    assert response.width == 0.06
    assert response.score == 0.91
    assert response.point_count == 42
    assert response.mask_pixel_count == 3
    assert response.segmentation_score == 0.87
