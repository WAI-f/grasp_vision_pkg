"""ROS2 camera subscriber with service-based SAM3 grasp pose estimation."""

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from threading import RLock
import gc
import time

from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Vector3
from grasp_vision_pkg.grasp_pose_estimator import GraspPoseEstimator
from grasp_vision_pkg.sam3_onnx_segmenter import SAM3OnnxSegmenter
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Float32


class CameraSubscriber(Node):
    """Subscribe to aligned RGB-D images and estimate grasp poses on request."""

    def __init__(self):
        super().__init__('camera_subscriber')

        self.bridge = CvBridge()
        self.color_camera_matrix = None
        self.aligned_depth_camera_matrix = None
        self.latest_color_frame = None
        self.latest_color_header = None
        self.latest_aligned_depth_frame = None
        self.latest_aligned_depth_header = None
        self.saved_color_image = False
        self.saved_aligned_depth_image = False
        self.saved_grasp_debug_image = False
        self.color_info_logged = False
        self.aligned_depth_info_logged = False
        self.sam3_segmenter = None
        self.grasp_estimator = None
        self.grasp_estimator_error = ''
        self.grasp_estimator_lock = RLock()

        self.declare_parameters(
            namespace='',
            parameters=[
                ('color_image_topic', '/color/image_raw'),
                ('color_info_topic', '/color/camera_info'),
                ('aligned_depth_image_topic', '/aligned_depth_to_color/image_raw'),
                ('aligned_depth_info_topic', '/aligned_depth_to_color/camera_info'),
                ('queue_size', 10),
                ('save_debug_images', True),
                ('save_dir', '/tmp/grasp_vision_images'),
                ('save_once', True),
                ('save_depth_preview', True),
                ('enable_grasp_pose', True),
                ('preload_grasp_estimator', True),
                ('grasp_pose_service', '/estimate_grasp_pose'),
                ('publish_grasp_result', False),
                ('save_grasp_debug_image', True),
                ('grasp_debug_image_name', 'grasp_pose_overlay.png'),
                ('grasp_pose_topic', '/grasp/pose'),
                ('grasp_width_topic', '/grasp/width'),
                ('grasp_debug_image_topic', '/grasp/debug_image'),
                ('segmented_cloud_stride', 2),
                ('background_mask_dilation_px', 5),
                ('background_object_bbox_filter', True),
                ('background_object_bbox_padding_xy', 0.06),
                ('background_object_bbox_padding_z', 0.04),
                ('min_object_dimension', 0.01),
                ('grasp_frame_id', ''),
                ('sam3_model_dir', ''),
                ('sam3_prompt_type', 'text'),
                ('sam3_prompt', 'visual'),
                ('sam3_box_prompt', [0.0, 0.0, 0.0, 0.0]),
                ('sam3_provider', 'CUDAExecutionProvider,CPUExecutionProvider'),
                ('sam3_input_width', 1008),
                ('sam3_input_height', 1008),
                ('sam3_warmup', True),
                ('sam3_score_threshold', 0.0),
                ('depth_scale', 0.001),
                ('float_depth_scale', 1.0),
                ('min_depth', 0.05),
                ('max_depth', 3.0),
                ('min_points', 80),
                ('max_points', 20000),
                ('outlier_percentile', 95.0),
                ('grasp_width_margin', 0.02),
            ],
        )

        color_image_topic = self.get_parameter('color_image_topic').value
        color_info_topic = self.get_parameter('color_info_topic').value
        aligned_depth_image_topic = self.get_parameter(
            'aligned_depth_image_topic'
        ).value
        aligned_depth_info_topic = self.get_parameter(
            'aligned_depth_info_topic'
        ).value
        queue_size = max(1, int(self.get_parameter('queue_size').value))
        self.save_debug_images = bool(self.get_parameter('save_debug_images').value)
        self.save_dir = Path(self.get_parameter('save_dir').value).expanduser()
        self.save_once = bool(self.get_parameter('save_once').value)
        self.save_depth_preview = bool(self.get_parameter('save_depth_preview').value)
        self.enable_grasp_pose = bool(self.get_parameter('enable_grasp_pose').value)
        self.preload_grasp_estimator = bool(
            self.get_parameter('preload_grasp_estimator').value
        )
        self.publish_grasp_result = bool(
            self.get_parameter('publish_grasp_result').value
        )
        self.save_grasp_debug_image = bool(
            self.get_parameter('save_grasp_debug_image').value
        )
        self.grasp_debug_image_name = str(
            self.get_parameter('grasp_debug_image_name').value
        )
        self.grasp_frame_id = str(self.get_parameter('grasp_frame_id').value)
        self.default_grasp_prompt = self._read_default_grasp_prompt()

        if self.save_debug_images:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(f'Debug images will be saved to {self.save_dir}')

        if self.enable_grasp_pose:
            grasp_pose_topic = self.get_parameter('grasp_pose_topic').value
            grasp_width_topic = self.get_parameter('grasp_width_topic').value
            grasp_debug_image_topic = self.get_parameter(
                'grasp_debug_image_topic'
            ).value
            self.segmented_cloud_stride = max(
                1, int(self.get_parameter('segmented_cloud_stride').value)
            )
            self.background_mask_dilation_px = max(
                0, int(self.get_parameter('background_mask_dilation_px').value)
            )
            self.background_object_bbox_filter = bool(
                self.get_parameter('background_object_bbox_filter').value
            )
            self.background_object_bbox_padding_xy = max(
                0.0,
                float(self.get_parameter('background_object_bbox_padding_xy').value),
            )
            self.background_object_bbox_padding_z = max(
                0.0,
                float(self.get_parameter('background_object_bbox_padding_z').value),
            )
            self.min_object_dimension = max(
                0.001, float(self.get_parameter('min_object_dimension').value)
            )
            grasp_pose_service = self.get_parameter('grasp_pose_service').value
            self.grasp_pose_pub = self.create_publisher(
                PoseStamped,
                grasp_pose_topic,
                queue_size,
            )
            self.grasp_width_pub = self.create_publisher(
                Float32,
                grasp_width_topic,
                queue_size,
            )
            self.grasp_debug_image_pub = self.create_publisher(
                Image,
                grasp_debug_image_topic,
                queue_size,
            )
            estimate_grasp_pose_srv = self._load_estimate_grasp_pose_service()
            self.grasp_pose_srv = self.create_service(
                estimate_grasp_pose_srv,
                grasp_pose_service,
                self.estimate_grasp_pose_callback,
            )
            self.get_logger().info(
                'Grasp pose service enabled with '
                f'service={grasp_pose_service}, '
                f'pose_topic={grasp_pose_topic}, '
                f'width_topic={grasp_width_topic}, '
                f'debug_image_topic={grasp_debug_image_topic}'
            )
        else:
            self.grasp_pose_pub = None
            self.grasp_width_pub = None
            self.grasp_debug_image_pub = None
            self.segmented_cloud_stride = 1
            self.background_mask_dilation_px = 0
            self.background_object_bbox_filter = False
            self.background_object_bbox_padding_xy = 0.0
            self.background_object_bbox_padding_z = 0.0
            self.min_object_dimension = 0.01

        self.color_image_sub = self.create_subscription(
            Image,
            color_image_topic,
            self.color_image_callback,
            queue_size,
        )
        self.color_info_sub = self.create_subscription(
            CameraInfo,
            color_info_topic,
            self.color_info_callback,
            queue_size,
        )
        self.aligned_depth_image_sub = self.create_subscription(
            Image,
            aligned_depth_image_topic,
            self.aligned_depth_image_callback,
            queue_size,
        )
        self.aligned_depth_info_sub = self.create_subscription(
            CameraInfo,
            aligned_depth_info_topic,
            self.aligned_depth_info_callback,
            queue_size,
        )

        self.get_logger().info(
            'Camera subscriber started with '
            f'color_image_topic={color_image_topic}, '
            f'color_info_topic={color_info_topic}, '
            f'aligned_depth_image_topic={aligned_depth_image_topic}, '
            f'aligned_depth_info_topic={aligned_depth_info_topic}, '
            f'queue_size={queue_size}'
        )

        if self.enable_grasp_pose and self.preload_grasp_estimator:
            self.get_logger().info(
                'Preloading grasp pose estimator and SAM3 ONNX models once.'
            )
            self._ensure_grasp_estimator()

    def _load_estimate_grasp_pose_service(self):
        try:
            from robot_interface_pkg.srv import EstimateGraspPose
        except ModuleNotFoundError as exc:
            message = (
                'EstimateGraspPose service type is unavailable from robot_interface_pkg. '
                'Build the package and source install/setup.bash before '
                'enabling grasp pose service.'
            )
            self.get_logger().error(message)
            raise RuntimeError(message) from exc
        return EstimateGraspPose

    def _create_sam3_segmenter(self):
        model_dir = self._resolve_sam3_model_dir()
        providers = self._split_csv_parameter(
            self.get_parameter('sam3_provider').value
        )
        input_width = int(self.get_parameter('sam3_input_width').value)
        input_height = int(self.get_parameter('sam3_input_height').value)
        warmup = bool(self.get_parameter('sam3_warmup').value)
        self.get_logger().info(
            'Loading SAM3 ONNX models once from '
            f'{model_dir} with providers={providers}'
        )
        segmenter = SAM3OnnxSegmenter(
            model_dir=model_dir,
            providers=providers,
            input_color_space='bgr',
            input_size=(input_width, input_height),
            score_threshold=float(
                self.get_parameter('sam3_score_threshold').value
            ),
            warmup=warmup,
        )
        self.get_logger().info('SAM3 ONNX models loaded and cached for reuse.')
        return segmenter

    def _ensure_sam3_segmenter(self):
        with self.grasp_estimator_lock:
            if self.sam3_segmenter is None:
                self.sam3_segmenter = self._create_sam3_segmenter()
            return self.sam3_segmenter

    def _create_grasp_estimator(self):
        segmenter = self._ensure_sam3_segmenter()
        return GraspPoseEstimator(
            segmenter=segmenter,
            depth_scale=float(self.get_parameter('depth_scale').value),
            float_depth_scale=float(self.get_parameter('float_depth_scale').value),
            min_depth=float(self.get_parameter('min_depth').value),
            max_depth=float(self.get_parameter('max_depth').value),
            min_points=int(self.get_parameter('min_points').value),
            max_points=int(self.get_parameter('max_points').value),
            outlier_percentile=float(
                self.get_parameter('outlier_percentile').value
            ),
            grasp_width_margin=float(
                self.get_parameter('grasp_width_margin').value
            ),
        )

    def _ensure_grasp_estimator(self):
        with self.grasp_estimator_lock:
            if self.grasp_estimator is not None:
                return True
            try:
                self.grasp_estimator = self._create_grasp_estimator()
                self.grasp_estimator_error = ''
                return True
            except Exception as exc:
                self.grasp_estimator_error = str(exc)
                self.get_logger().error(
                    f'Failed to initialize grasp pose estimator: {exc}'
                )
                return False

    def _resolve_sam3_model_dir(self):
        raw_model_dir = str(self.get_parameter('sam3_model_dir').value).strip()
        if raw_model_dir:
            return str(Path(raw_model_dir).expanduser())

        try:
            share_dir = Path(get_package_share_directory('grasp_vision_pkg'))
            return str(share_dir / 'models' / 'sam3')
        except Exception:
            source_root = Path(__file__).resolve().parents[1]
            return str(source_root / 'models' / 'sam3')

    def _read_default_grasp_prompt(self):
        prompt_type = str(self.get_parameter('sam3_prompt_type').value).lower().strip()
        text_prompt = str(self.get_parameter('sam3_prompt').value).strip() or 'visual'
        return self._make_prompt(prompt_type, text_prompt, self._read_box_prompt_param())

    def _read_box_prompt_param(self):
        return [float(value) for value in self.get_parameter('sam3_box_prompt').value]

    def _make_prompt(self, prompt_type, text_prompt, box_prompt):
        if prompt_type == 'text':
            return text_prompt or 'visual'
        if prompt_type == 'box':
            box = [float(value) for value in box_prompt]
            if len(box) != 4:
                raise ValueError('box prompt must contain 4 values')
            return {
                'type': 'box',
                'value': box,
                'text_prompt': text_prompt or 'visual',
            }
        raise ValueError("prompt_type must be 'text' or 'box'")

    def _prompt_from_request(self, request):
        if request.use_default_prompt:
            return self.default_grasp_prompt
        prompt_type = str(request.prompt_type).lower().strip()
        text_prompt = str(request.prompt).strip() or 'visual'
        return self._make_prompt(prompt_type, text_prompt, list(request.box_prompt))

    def _split_csv_parameter(self, raw_value):
        return [
            part.strip()
            for part in str(raw_value).split(',')
            if part.strip()
        ]

    def color_info_callback(self, msg: CameraInfo):
        self.color_camera_matrix = msg.k
        if not self.color_info_logged:
            self.get_logger().info(f'Received color camera info K={list(msg.k)}')
            self.color_info_logged = True

    def aligned_depth_info_callback(self, msg: CameraInfo):
        self.aligned_depth_camera_matrix = msg.k
        if not self.aligned_depth_info_logged:
            self.get_logger().info(
                f'Received aligned depth camera info K={list(msg.k)}'
            )
            self.aligned_depth_info_logged = True

    def color_image_callback(self, msg: Image):
        color_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.latest_color_frame = color_frame
        self.latest_color_header = msg.header

        if self._should_save(self.saved_color_image):
            timestamp = self._timestamp_from_msg(msg)
            image_path = self.save_dir / f'color_{timestamp}.png'
            self._write_image(image_path, color_frame)
            self.saved_color_image = True

        self.get_logger().info(
            f'Received color image: {msg.width}x{msg.height}'
        )

    def aligned_depth_image_callback(self, msg: Image):
        aligned_depth_frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='passthrough',
        )
        self.latest_aligned_depth_frame = aligned_depth_frame
        self.latest_aligned_depth_header = msg.header

        if self._should_save(self.saved_aligned_depth_image):
            timestamp = self._timestamp_from_msg(msg)
            self._save_depth_image(aligned_depth_frame, msg.encoding, timestamp)
            self.saved_aligned_depth_image = True

        self.get_logger().info(
            'Received aligned depth image: '
            f'{msg.width}x{msg.height}, encoding={msg.encoding}'
        )

    def estimate_grasp_pose_callback(self, request, response):
        start_time = time.monotonic()
        response = self._estimate_grasp_pose(request, response)
        self._fill_processing_time(response, time.monotonic() - start_time)
        return response

    def _estimate_grasp_pose(self, request, response):
        snapshot = self._latest_rgbd_snapshot()
        if snapshot is None:
            return self._set_failure(
                response,
                response.STATUS_NOT_READY,
                'Waiting for color image, aligned depth image, and camera info.',
            )
        color_frame, depth_frame, camera_matrix, source_header = snapshot

        try:
            prompt = self._prompt_from_request(request)
        except Exception as exc:
            return self._set_failure(
                response,
                response.STATUS_INVALID_REQUEST,
                f'Invalid grasp prompt: {exc}',
            )

        if not self._ensure_grasp_estimator():
            return self._set_failure(
                response,
                response.STATUS_ESTIMATOR_UNAVAILABLE,
                'Estimator unavailable: ' + self.grasp_estimator_error,
            )

        try:
            pose = self.grasp_estimator.estimate(
                rgb_image=color_frame,
                aligned_depth_image=depth_frame,
                camera_matrix=camera_matrix,
                object_prompt=prompt,
            )
        except Exception as exc:
            response = self._set_failure(
                response,
                response.STATUS_ESTIMATION_FAILED,
                f'Failed to estimate grasp pose: {exc}',
            )
            self._release_grasp_inference_cache()
            return response

        response.success = True
        response.status_code = response.STATUS_SUCCESS
        response.message = 'Grasp pose estimated.'
        response.pose = self._pose_to_msg(pose, source_header)
        object_pose, object_dimensions = self._object_bounds_to_msgs(
            self.grasp_estimator.last_points,
            source_header,
        )
        response.object_pose = object_pose
        response.object_dimensions = object_dimensions
        response.width = float(pose.width)
        response.score = float(pose.score)
        response.point_count = int(pose.point_count)
        response.mask_pixel_count = self._last_mask_pixel_count()
        response.segmentation_score = self._last_segmentation_score()

        object_cloud = None
        background_cloud = None
        if self.grasp_estimator.last_mask is not None:
            try:
                object_cloud, background_cloud = self._make_segmented_clouds(
                    color_frame,
                    depth_frame,
                    camera_matrix,
                    self.grasp_estimator.last_mask,
                    source_header,
                )
                response.object_cloud = deepcopy(object_cloud)
                response.background_cloud = deepcopy(background_cloud)
                response.has_segmented_clouds = True
            except Exception as exc:
                response.has_segmented_clouds = False
                self.get_logger().warn(
                    f'Failed to build segmented point clouds for response: {exc}'
                )
        else:
            response.has_segmented_clouds = False

        if bool(request.publish_result) or self.publish_grasp_result:
            self._publish_grasp_pose_msg(response.pose)
            self._publish_grasp_width(pose.width)
            self._publish_grasp_debug_image(pose, camera_matrix, source_header)

        if bool(request.save_debug_image) or self.save_grasp_debug_image:
            response.debug_image_path = self._save_grasp_debug_image(
                pose,
                camera_matrix,
            )

        self.get_logger().info(
            'Service estimated grasp pose: '
            f'position={pose.position.tolist()}, '
            f'quaternion_xyzw={pose.quaternion_xyzw.tolist()}, '
            f'width={pose.width:.4f}m, '
            f'score={pose.score:.3f}, '
            f'point_count={pose.point_count}'
        )
        self._release_grasp_inference_cache()
        return response

    def _latest_rgbd_snapshot(self):
        if self.latest_color_frame is None or self.latest_aligned_depth_frame is None:
            return None
        if self.color_camera_matrix is not None:
            camera_matrix = self.color_camera_matrix
        else:
            camera_matrix = self.aligned_depth_camera_matrix
        if camera_matrix is None:
            return None
        source_header = self.latest_color_header or self.latest_aligned_depth_header
        return (
            self.latest_color_frame.copy(),
            self.latest_aligned_depth_frame.copy(),
            np.asarray(camera_matrix, dtype=np.float64).copy(),
            source_header,
        )

    def _set_failure(self, response, status_code, message):
        response.success = False
        response.status_code = status_code
        response.message = message
        self.get_logger().warn(message)
        return response

    def _fill_processing_time(self, response, elapsed_seconds):
        seconds = int(elapsed_seconds)
        response.processing_time.sec = seconds
        response.processing_time.nanosec = int((elapsed_seconds - seconds) * 1e9)

    def _pose_to_msg(self, pose, source_header):
        msg = PoseStamped()
        self._fill_output_header(msg.header, source_header)
        msg.pose.position.x = float(pose.position[0])
        msg.pose.position.y = float(pose.position[1])
        msg.pose.position.z = float(pose.position[2])
        msg.pose.orientation.x = float(pose.quaternion_xyzw[0])
        msg.pose.orientation.y = float(pose.quaternion_xyzw[1])
        msg.pose.orientation.z = float(pose.quaternion_xyzw[2])
        msg.pose.orientation.w = float(pose.quaternion_xyzw[3])
        return msg

    def _last_mask_pixel_count(self):
        if self.grasp_estimator is None or self.grasp_estimator.last_mask is None:
            return 0
        return int(np.count_nonzero(self.grasp_estimator.last_mask))

    def _last_segmentation_score(self):
        if self.grasp_estimator is None:
            return float('nan')
        segmenter = getattr(self.grasp_estimator, 'segmenter', None)
        prediction = getattr(segmenter, 'last_prediction', None)
        if prediction is None:
            return float('nan')
        return float(prediction.scores[prediction.selected_index])

    def _release_grasp_inference_cache(self):
        estimator = self.grasp_estimator
        segmenter = self.sam3_segmenter
        if estimator is not None:
            estimator.last_mask = None
            estimator.last_points = None
            segmenter = getattr(estimator, 'segmenter', segmenter)

        if segmenter is not None:
            if hasattr(segmenter, 'last_prediction'):
                segmenter.last_prediction = None
            if hasattr(segmenter, 'last_mask'):
                segmenter.last_mask = None

        gc.collect()
        self._trim_native_heap()

    @staticmethod
    def _trim_native_heap():
        try:
            import ctypes

            malloc_trim = getattr(ctypes.CDLL('libc.so.6'), 'malloc_trim', None)
            if malloc_trim is not None:
                malloc_trim(0)
        except Exception:
            pass

    def _object_bounds_to_msgs(self, points, source_header):
        pose_msg = PoseStamped()
        self._fill_output_header(pose_msg.header, source_header)
        pose_msg.pose.orientation.w = 1.0

        dimensions = Vector3()
        if points is None or len(points) == 0:
            return pose_msg, dimensions

        point_array = np.asarray(points, dtype=np.float64)
        mins = np.nanmin(point_array, axis=0)
        maxs = np.nanmax(point_array, axis=0)
        center = (mins + maxs) * 0.5
        size = np.maximum(maxs - mins, float(self.min_object_dimension))

        pose_msg.pose.position.x = float(center[0])
        pose_msg.pose.position.y = float(center[1])
        pose_msg.pose.position.z = float(center[2])
        dimensions.x = float(size[0])
        dimensions.y = float(size[1])
        dimensions.z = float(size[2])
        return pose_msg, dimensions

    def _make_segmented_clouds(
        self,
        color_frame,
        depth_frame,
        camera_matrix,
        object_mask,
        source_header,
    ):
        intrinsics = self.grasp_estimator._normalize_camera_matrix(camera_matrix)
        depth_m = self.grasp_estimator._depth_to_meters(depth_frame)
        mask = np.asarray(object_mask).astype(bool)
        if mask.shape != depth_m.shape:
            raise ValueError(
                f'Object mask shape {mask.shape} does not match depth shape '
                f'{depth_m.shape}'
            )

        valid = (
            np.isfinite(depth_m)
            & (depth_m >= float(self.grasp_estimator.min_depth))
            & (depth_m <= float(self.grasp_estimator.max_depth))
        )
        object_selector = valid & mask
        background_mask = self._dilate_mask(mask, self.background_mask_dilation_px)
        object_volume_mask = self._object_bbox_volume_mask(
            valid,
            object_selector,
            depth_m,
            intrinsics,
        )
        background_selector = valid & ~background_mask & ~object_volume_mask
        header = self._cloud_header(source_header)
        return (
            self._selector_to_cloud(
                object_selector,
                depth_m,
                color_frame,
                intrinsics,
                header,
            ),
            self._selector_to_cloud(
                background_selector,
                depth_m,
                color_frame,
                intrinsics,
                header,
            ),
        )

    @staticmethod
    def _dilate_mask(mask, dilation_px):
        if dilation_px <= 0:
            return mask
        kernel_size = dilation_px * 2 + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)

    def _object_bbox_volume_mask(self, valid, object_selector, depth_m, intrinsics):
        if not self.background_object_bbox_filter or not object_selector.any():
            return np.zeros_like(valid, dtype=bool)

        object_points = self._selector_xyz(object_selector, depth_m, intrinsics)
        if object_points.size == 0:
            return np.zeros_like(valid, dtype=bool)

        mins = np.nanmin(object_points, axis=0)
        maxs = np.nanmax(object_points, axis=0)
        padding = np.array(
            [
                self.background_object_bbox_padding_xy,
                self.background_object_bbox_padding_xy,
                self.background_object_bbox_padding_z,
            ],
            dtype=np.float32,
        )
        mins = mins - padding
        maxs = maxs + padding

        rows, cols = np.nonzero(valid)
        if rows.size == 0:
            return np.zeros_like(valid, dtype=bool)

        points = self._pixels_to_xyz(rows, cols, depth_m, intrinsics)
        inside = np.all((points >= mins) & (points <= maxs), axis=1)
        volume_mask = np.zeros_like(valid, dtype=bool)
        volume_mask[rows[inside], cols[inside]] = True

        removed_points = int(np.count_nonzero(volume_mask & ~object_selector))
        self.get_logger().info(
            'Background object-volume filter: '
            f'bounds_min={mins.tolist()}, bounds_max={maxs.tolist()}, '
            f'removed_background_points={removed_points}'
        )
        return volume_mask

    def _selector_xyz(self, selector, depth_m, intrinsics):
        rows, cols = np.nonzero(selector)
        if rows.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        return self._pixels_to_xyz(rows, cols, depth_m, intrinsics)

    @staticmethod
    def _pixels_to_xyz(rows, cols, depth_m, intrinsics):
        z = depth_m[rows, cols].astype(np.float32)
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        x = ((cols.astype(np.float32) - cx) / fx) * z
        y = ((rows.astype(np.float32) - cy) / fy) * z
        return np.column_stack((x, y, z)).astype(np.float32, copy=False)

    def _cloud_header(self, source_header):
        if source_header is not None:
            return deepcopy(source_header)
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        return msg.header

    def _selector_to_cloud(
        self,
        selector,
        depth_m,
        color_frame,
        intrinsics,
        header,
    ):
        rows, cols = np.nonzero(selector)
        stride = int(self.segmented_cloud_stride)
        if stride > 1 and rows.size > 0:
            keep = (rows % stride == 0) & (cols % stride == 0)
            rows = rows[keep]
            cols = cols[keep]

        msg = PointCloud2()
        msg.header = deepcopy(header)
        msg.height = 1
        msg.is_bigendian = False
        msg.is_dense = True
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 16

        if rows.size == 0:
            msg.width = 0
            msg.row_step = 0
            msg.data = b''
            return msg

        xyz = self._pixels_to_xyz(rows, cols, depth_m, intrinsics)
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        colors = color_frame[rows, cols]
        if colors.ndim == 1:
            colors = np.repeat(colors[:, None], 3, axis=1)
        bgr = colors[:, :3].astype(np.uint32)
        rgb_uint32 = (
            (bgr[:, 2] << 16)
            | (bgr[:, 1] << 8)
            | bgr[:, 0]
        )

        cloud = np.empty(
            rows.size,
            dtype=[
                ('x', '<f4'),
                ('y', '<f4'),
                ('z', '<f4'),
                ('rgb', '<f4'),
            ],
        )
        cloud['x'] = x
        cloud['y'] = y
        cloud['z'] = z
        cloud['rgb'] = rgb_uint32.view(np.float32)

        msg.width = rows.size
        msg.row_step = msg.point_step * msg.width
        msg.data = cloud.tobytes()
        return msg

    def _publish_grasp_pose_msg(self, msg):
        if self.grasp_pose_pub is not None:
            self.grasp_pose_pub.publish(msg)

    def _publish_grasp_width(self, width):
        if self.grasp_width_pub is None:
            return
        msg = Float32()
        msg.data = float(width)
        self.grasp_width_pub.publish(msg)

    def _publish_grasp_debug_image(self, pose, camera_matrix, source_header):
        if self.grasp_debug_image_pub is None:
            return
        overlay = self.grasp_estimator.visualize(
            self.latest_color_frame,
            pose,
            camera_matrix,
        )
        msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        self._fill_output_header(msg.header, source_header)
        self.grasp_debug_image_pub.publish(msg)

    def _save_grasp_debug_image(self, pose, camera_matrix):
        if self.save_once and self.saved_grasp_debug_image:
            return str(self.save_dir / str(self.grasp_debug_image_name))
        overlay = self.grasp_estimator.visualize(
            self.latest_color_frame,
            pose,
            camera_matrix,
        )
        image_path = self.save_dir / str(self.grasp_debug_image_name)
        self._write_image(image_path, overlay)
        self.saved_grasp_debug_image = True
        return str(image_path)

    def _fill_output_header(self, header, source_header=None):
        if source_header is not None:
            header.stamp = source_header.stamp
            header.frame_id = source_header.frame_id
        else:
            header.stamp = self.get_clock().now().to_msg()
        if self.grasp_frame_id:
            header.frame_id = self.grasp_frame_id

    def _should_save(self, already_saved):
        if not self.save_debug_images:
            return False
        return not (self.save_once and already_saved)

    def _timestamp_from_msg(self, msg: Image):
        stamp = msg.header.stamp
        if stamp.sec != 0 or stamp.nanosec != 0:
            return f'{stamp.sec}_{stamp.nanosec:09d}'
        return datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    def _save_depth_image(self, depth_frame, encoding, timestamp):
        if np.issubdtype(depth_frame.dtype, np.floating):
            depth_path = self.save_dir / f'aligned_depth_{timestamp}.npy'
            np.save(depth_path, depth_frame)
            self.get_logger().info(
                f'Saved aligned depth data ({encoding}) to {depth_path}'
            )
        else:
            depth_path = self.save_dir / f'aligned_depth_{timestamp}.png'
            self._write_image(depth_path, depth_frame)

        if self.save_depth_preview:
            preview = self._create_depth_preview(depth_frame)
            preview_path = self.save_dir / f'aligned_depth_preview_{timestamp}.png'
            self._write_image(preview_path, preview)

    def _create_depth_preview(self, depth_frame):
        depth_array = np.asarray(depth_frame)
        finite_mask = np.isfinite(depth_array)
        if not finite_mask.any():
            return np.zeros(depth_array.shape, dtype=np.uint8)

        finite_values = depth_array[finite_mask]
        min_depth = float(finite_values.min())
        max_depth = float(finite_values.max())
        if max_depth <= min_depth:
            return np.zeros(depth_array.shape, dtype=np.uint8)

        sanitized_depth = np.nan_to_num(
            depth_array,
            nan=min_depth,
            posinf=max_depth,
            neginf=min_depth,
        )
        preview = (sanitized_depth - min_depth) * (255.0 / (max_depth - min_depth))
        return np.clip(preview, 0, 255).astype(np.uint8)

    def _write_image(self, image_path, image):
        if cv2.imwrite(str(image_path), image):
            self.get_logger().info(f'Saved image to {image_path}')
        else:
            self.get_logger().error(f'Failed to save image to {image_path}')


def main(args=None):
    """Run the camera subscriber node."""
    rclpy.init(args=args)
    node = CameraSubscriber()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
