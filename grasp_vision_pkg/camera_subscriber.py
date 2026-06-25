from datetime import datetime
from pathlib import Path

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class CameraSubscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')

        self.bridge = CvBridge()
        self.color_camera_matrix = None
        self.aligned_depth_camera_matrix = None
        self.latest_color_frame = None
        self.latest_aligned_depth_frame = None
        self.saved_color_image = False
        self.saved_aligned_depth_image = False
        self.color_info_logged = False
        self.aligned_depth_info_logged = False

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
            ],
        )

        color_image_topic = self.get_parameter('color_image_topic').value
        color_info_topic = self.get_parameter('color_info_topic').value
        aligned_depth_image_topic = self.get_parameter('aligned_depth_image_topic').value
        aligned_depth_info_topic = self.get_parameter('aligned_depth_info_topic').value
        queue_size = max(1, int(self.get_parameter('queue_size').value))
        self.save_debug_images = bool(self.get_parameter('save_debug_images').value)
        self.save_dir = Path(self.get_parameter('save_dir').value).expanduser()
        self.save_once = bool(self.get_parameter('save_once').value)
        self.save_depth_preview = bool(self.get_parameter('save_depth_preview').value)

        if self.save_debug_images:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(f'Debug images will be saved to {self.save_dir}')

        # 订阅彩色图像
        self.color_image_sub = self.create_subscription(
            Image,
            color_image_topic,
            self.color_image_callback,
            queue_size
        )

        # 订阅彩色相机内参
        self.color_info_sub = self.create_subscription(
            CameraInfo,
            color_info_topic,
            self.color_info_callback,
            queue_size
        )

        # 订阅对齐到彩色图像坐标系的深度图像
        self.aligned_depth_image_sub = self.create_subscription(
            Image,
            aligned_depth_image_topic,
            self.aligned_depth_image_callback,
            queue_size
        )

        # 订阅对齐深度相机内参
        self.aligned_depth_info_sub = self.create_subscription(
            CameraInfo,
            aligned_depth_info_topic,
            self.aligned_depth_info_callback,
            queue_size
        )

        self.get_logger().info(
            'Camera subscriber started with '
            f'color_image_topic={color_image_topic}, '
            f'color_info_topic={color_info_topic}, '
            f'aligned_depth_image_topic={aligned_depth_image_topic}, '
            f'aligned_depth_info_topic={aligned_depth_info_topic}, '
            f'queue_size={queue_size}'
        )

    def color_info_callback(self, msg: CameraInfo):
        self.color_camera_matrix = msg.k  # 3x3 intrinsic
        if not self.color_info_logged:
            self.get_logger().info(f'Received color camera info K={list(msg.k)}')
            self.color_info_logged = True

    def aligned_depth_info_callback(self, msg: CameraInfo):
        self.aligned_depth_camera_matrix = msg.k  # 3x3 intrinsic
        if not self.aligned_depth_info_logged:
            self.get_logger().info(
                f'Received aligned depth camera info K={list(msg.k)}'
            )
            self.aligned_depth_info_logged = True

    def color_image_callback(self, msg: Image):

        # ROS Image -> OpenCV
        color_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.latest_color_frame = color_frame

        if self._should_save(self.saved_color_image):
            timestamp = self._timestamp_from_msg(msg)
            image_path = self.save_dir / f'color_{timestamp}.png'
            self._write_image(image_path, color_frame)
            self.saved_color_image = True

        # 显示图像
        # cv2.imshow('color', color_frame)
        # cv2.waitKey(1)

        self.get_logger().info(
            f'Received color image: {msg.width}x{msg.height}'
        )

    def aligned_depth_image_callback(self, msg: Image):

        # ROS depth Image -> OpenCV, keep original depth encoding (16UC1/32FC1)
        aligned_depth_frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding='passthrough'
        )
        self.latest_aligned_depth_frame = aligned_depth_frame
        # print(aligned_depth_frame[300,400])

        if self._should_save(self.saved_aligned_depth_image):
            timestamp = self._timestamp_from_msg(msg)
            self._save_depth_image(aligned_depth_frame, msg.encoding, timestamp)
            self.saved_aligned_depth_image = True

        # 显示归一化后的深度图，便于调试
        depth_display = self._create_depth_preview(aligned_depth_frame)
        # cv2.imshow('aligned_depth', depth_display)
        # cv2.waitKey(1)

        self.get_logger().info(
            f'Received aligned depth image: {msg.width}x{msg.height}, encoding={msg.encoding}'
        )

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
            neginf=min_depth
        )
        preview = (sanitized_depth - min_depth) * (255.0 / (max_depth - min_depth))
        return np.clip(preview, 0, 255).astype(np.uint8)

    def _write_image(self, image_path, image):
        if cv2.imwrite(str(image_path), image):
            self.get_logger().info(f'Saved image to {image_path}')
        else:
            self.get_logger().error(f'Failed to save image to {image_path}')


def main(args=None):
    rclpy.init(args=args)

    node = CameraSubscriber()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()
