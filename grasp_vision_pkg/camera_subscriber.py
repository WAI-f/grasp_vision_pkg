"""ROS2 camera subscriber with service-based SAM3 grasp pose estimation."""

from datetime import datetime
from pathlib import Path
import time

from ament_index_python.packages import get_package_share_directory
import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from grasp_vision_pkg.grasp_pose_estimator import GraspPoseEstimator
from grasp_vision_pkg.sam3_onnx_segmenter import SAM3OnnxSegmenter
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
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
        self.grasp_estimator = None
        self.grasp_estimator_error = ''

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
                ('grasp_pose_service', '/estimate_grasp_pose'),
                ('publish_grasp_result', False),
                ('save_grasp_debug_image', True),
                ('grasp_debug_image_name', 'grasp_pose_overlay.png'),
                ('grasp_pose_topic', '/grasp/pose'),
                ('grasp_width_topic', '/grasp/width'),
                ('grasp_debug_image_topic', '/grasp/debug_image'),
                ('grasp_frame_id', ''),
                ('sam3_model_dir', ''),
                ('sam3_prompt_type', 'text'),
                ('sam3_prompt', 'visual'),
                ('sam3_box_prompt', [0.0, 0.0, 0.0, 0.0]),
                ('sam3_provider', 'CUDAExecutionProvider,CPUExecutionProvider'),
                ('sam3_input_width', 1008),
                ('sam3_input_height', 1008),
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
            self.grasp_pose_srv = None

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

    def _create_grasp_estimator(self):
        model_dir = self._resolve_sam3_model_dir()
        providers = self._split_csv_parameter(
            self.get_parameter('sam3_provider').value
        )
        input_width = int(self.get_parameter('sam3_input_width').value)
        input_height = int(self.get_parameter('sam3_input_height').value)
        segmenter = SAM3OnnxSegmenter(
            model_dir=model_dir,
            providers=providers,
            input_color_space='bgr',
            input_size=(input_width, input_height),
            score_threshold=float(
                self.get_parameter('sam3_score_threshold').value
            ),
        )
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
            return self._set_failure(
                response,
                response.STATUS_ESTIMATION_FAILED,
                f'Failed to estimate grasp pose: {exc}',
            )

        response.success = True
        response.status_code = response.STATUS_SUCCESS
        response.message = 'Grasp pose estimated.'
        response.pose = self._pose_to_msg(pose, source_header)
        response.width = float(pose.width)
        response.score = float(pose.score)
        response.point_count = int(pose.point_count)
        response.mask_pixel_count = self._last_mask_pixel_count()
        response.segmentation_score = self._last_segmentation_score()

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
