import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2


class CameraSubscriber(Node):

    def __init__(self):
        super().__init__('camera_subscriber')

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.latest_depth_frame = None

        # 订阅 RGB 图像
        self.image_sub = self.create_subscription(
            Image,
            '/isaac_rgb',   # 根据你的相机topic改
            self.image_callback,
            10
        )

        # 订阅深度图像
        self.depth_sub = self.create_subscription(
            Image,
            '/isaac_depth',   # 根据你的相机topic改
            self.depth_callback,
            10
        )

        # （可选）订阅相机内参
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/isaac_camera_info',
            self.info_callback,
            10
        )

        self.get_logger().info("Camera subscriber started")

    def info_callback(self, msg: CameraInfo):
        self.camera_matrix = msg.k  # 3x3 intrinsic
        self.get_logger().info("Received camera info")

    def image_callback(self, msg: Image):

        # ROS Image -> OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # 显示图像
        # cv2.imshow("camera", frame)
        # cv2.waitKey(1)

        self.get_logger().info(
            f"Received image: {msg.width}x{msg.height}"
        )

    def depth_callback(self, msg: Image):

        # ROS depth Image -> OpenCV, keep original depth encoding (16UC1/32FC1)
        depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.latest_depth_frame = depth_frame

        # # 显示归一化后的深度图，便于调试
        # depth_display = cv2.normalize(
        #     depth_frame,
        #     None,
        #     alpha=0,
        #     beta=255,
        #     norm_type=cv2.NORM_MINMAX,
        #     dtype=cv2.CV_8U
        # )
        # cv2.imshow("depth", depth_display)
        # cv2.waitKey(1)

        self.get_logger().info(
            f"Received depth image: {msg.width}x{msg.height}, encoding={msg.encoding}"
        )


def main(args=None):
    rclpy.init(args=args)

    node = CameraSubscriber()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()