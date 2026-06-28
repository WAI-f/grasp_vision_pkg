"""Estimate a simple grasp pose from RGB and aligned depth images.

A concrete SAM3 ONNX backend lives in ``sam3_onnx_segmenter.py`` and can be
plugged into ``GraspPoseEstimator`` when SAM3-based segmentation is needed.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class GraspPose:
    """Grasp pose in the aligned color camera frame.

    ``orientation_matrix`` columns are the gripper frame axes expressed in the
    camera frame: x is the finger closing axis, y is the object long axis, and
    z points from the object back toward the camera.
    """

    position: np.ndarray
    orientation_matrix: np.ndarray
    quaternion_xyzw: np.ndarray
    width: float
    score: float
    center_pixel: Tuple[int, int]
    point_count: int


class SAM3SegmenterInterface(Protocol):
    """Interface expected from a SAM3 object segmentation backend."""

    def segment(self, rgb_image: np.ndarray, prompt: Any = None) -> np.ndarray:
        """Return a boolean object mask with shape ``H x W``.

        ``prompt`` can later be a text label, point prompt, box prompt, or any
        SAM3-specific object query. This project does not implement SAM3 here.
        """
        raise NotImplementedError


class GraspPoseEstimator:
    """Estimate a PCA-based grasp pose from an object mask and aligned depth."""

    def __init__(
        self,
        segmenter: Optional[SAM3SegmenterInterface] = None,
        depth_scale: float = 0.001,
        float_depth_scale: float = 1.0,
        min_depth: float = 0.05,
        max_depth: float = 3.0,
        min_points: int = 80,
        max_points: int = 20000,
        outlier_percentile: float = 95.0,
        grasp_width_margin: float = 0.02,
    ):
        self.segmenter = segmenter
        self.depth_scale = depth_scale
        self.float_depth_scale = float_depth_scale
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_points = min_points
        self.max_points = max_points
        self.outlier_percentile = outlier_percentile
        self.grasp_width_margin = grasp_width_margin
        self.last_mask = None
        self.last_points = None

    def estimate(
        self,
        rgb_image: np.ndarray,
        aligned_depth_image: np.ndarray,
        camera_matrix: Any,
        object_prompt: Any = None,
        object_mask: Optional[np.ndarray] = None,
    ) -> GraspPose:
        """Estimate and return one grasp pose.

        Args:
            rgb_image: Color image aligned with ``aligned_depth_image``.
            aligned_depth_image: Depth image in the color image coordinate
                frame. Integer depth is treated as millimeters by default;
                floating-point depth is treated as meters by default.
            camera_matrix: 3x3 intrinsic matrix, flattened length-9 matrix, or
                an object with ROS ``CameraInfo.k`` style data.
            object_prompt: Prompt forwarded to the SAM3 segmenter when
                ``object_mask`` is not provided.
            object_mask: Optional precomputed object mask. Supplying this lets
                callers test the PCA and pose logic before SAM3 is available.
        """
        self._validate_images(rgb_image, aligned_depth_image)
        intrinsics = self._normalize_camera_matrix(camera_matrix)
        mask = self._get_object_mask(rgb_image, object_prompt, object_mask)
        points, pixels = self._object_points(mask, aligned_depth_image, intrinsics)

        if points.shape[0] < self.min_points:
            raise ValueError(
                f'Not enough valid object depth points: {points.shape[0]} '
                f'< {self.min_points}'
            )

        points = self._remove_outliers(points)
        pose = self._estimate_pose_from_points(points, pixels)
        self.last_mask = mask
        self.last_points = points
        return pose

    def visualize(
        self,
        rgb_image: np.ndarray,
        grasp_pose: GraspPose,
        camera_matrix: Any,
        object_mask: Optional[np.ndarray] = None,
        axis_length: float = 0.08,
    ) -> np.ndarray:
        """Return an image with mask overlay and projected grasp axes."""
        intrinsics = self._normalize_camera_matrix(camera_matrix)
        vis = rgb_image.copy()

        if object_mask is None:
            object_mask = self.last_mask
        if object_mask is not None:
            vis = self._overlay_mask(vis, object_mask)

        center = grasp_pose.position
        rotation = grasp_pose.orientation_matrix
        center_uv = self._project_point(center, intrinsics)

        if center_uv is not None:
            cv2.circle(vis, center_uv, 5, (0, 255, 255), -1)

        half_width = max(grasp_pose.width * 0.5, axis_length * 0.35)
        self._draw_3d_segment(
            vis,
            center - rotation[:, 0] * half_width,
            center + rotation[:, 0] * half_width,
            intrinsics,
            (255, 255, 0),
            2,
        )
        self._draw_axis(
            vis,
            center,
            rotation[:, 0],
            intrinsics,
            axis_length,
            (0, 0, 255),
        )
        self._draw_axis(
            vis,
            center,
            rotation[:, 1],
            intrinsics,
            axis_length,
            (0, 255, 0),
        )
        self._draw_axis(
            vis,
            center,
            rotation[:, 2],
            intrinsics,
            axis_length,
            (255, 0, 0),
        )

        cv2.putText(
            vis,
            f'width={grasp_pose.width:.3f}m score={grasp_pose.score:.2f}',
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return vis

    def _get_object_mask(
        self,
        rgb_image: np.ndarray,
        object_prompt: Any,
        object_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        if object_mask is None:
            if self.segmenter is None:
                raise ValueError(
                    'object_mask is required until a SAM3 segmenter is provided'
                )
            object_mask = self.segmenter.segment(rgb_image, object_prompt)

        mask = np.asarray(object_mask).astype(bool)
        if mask.shape[:2] != rgb_image.shape[:2]:
            raise ValueError(
                f'Object mask shape {mask.shape} does not match image shape '
                f'{rgb_image.shape[:2]}'
            )
        return mask

    def _object_points(
        self,
        mask: np.ndarray,
        depth_image: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        depth_m = self._depth_to_meters(depth_image)
        valid = (
            mask
            & np.isfinite(depth_m)
            & (depth_m >= self.min_depth)
            & (depth_m <= self.max_depth)
        )
        rows, cols = np.nonzero(valid)
        if rows.size == 0:
            return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=int)

        if rows.size > self.max_points:
            indices = np.linspace(0, rows.size - 1, self.max_points).astype(int)
            rows = rows[indices]
            cols = cols[indices]

        z = depth_m[rows, cols].astype(np.float64)
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        x = (cols.astype(np.float64) - cx) * z / fx
        y = (rows.astype(np.float64) - cy) * z / fy
        points = np.column_stack((x, y, z))
        pixels = np.column_stack((cols, rows))
        return points, pixels

    def _estimate_pose_from_points(
        self,
        points: np.ndarray,
        pixels: np.ndarray,
    ) -> GraspPose:
        center = np.median(points, axis=0)
        centered = points - center
        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        long_axis = self._normalize(eigenvectors[:, 0])
        normal_axis = self._normalize(eigenvectors[:, 2])
        if normal_axis[2] > 0.0:
            normal_axis = -normal_axis

        long_axis = long_axis - normal_axis * np.dot(long_axis, normal_axis)
        long_axis = self._normalize(long_axis)
        closing_axis = self._normalize(np.cross(long_axis, normal_axis))
        long_axis = self._normalize(np.cross(normal_axis, closing_axis))

        rotation = np.column_stack((closing_axis, long_axis, normal_axis))
        width = self._estimate_grasp_width(points, center, closing_axis)
        center_pixel = tuple(np.round(np.median(pixels, axis=0)).astype(int))
        score = self._estimate_score(eigenvalues, points.shape[0])
        quaternion = self._rotation_matrix_to_quaternion(rotation)

        return GraspPose(
            position=center,
            orientation_matrix=rotation,
            quaternion_xyzw=quaternion,
            width=width,
            score=score,
            center_pixel=center_pixel,
            point_count=points.shape[0],
        )

    def _estimate_grasp_width(
        self,
        points: np.ndarray,
        center: np.ndarray,
        closing_axis: np.ndarray,
    ) -> float:
        coordinates = (points - center) @ closing_axis
        low, high = np.percentile(coordinates, [5.0, 95.0])
        return float(max(0.0, high - low) + self.grasp_width_margin)

    def _estimate_score(self, eigenvalues: np.ndarray, point_count: int) -> float:
        anisotropy = (eigenvalues[0] - eigenvalues[-1]) / max(eigenvalues[0], 1e-9)
        density = min(1.0, point_count / max(float(self.min_points) * 5.0, 1.0))
        return float(np.clip(0.6 * anisotropy + 0.4 * density, 0.0, 1.0))

    def _remove_outliers(self, points: np.ndarray) -> np.ndarray:
        if points.shape[0] < self.min_points * 2:
            return points
        center = np.median(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        threshold = np.percentile(distances, self.outlier_percentile)
        filtered = points[distances <= threshold]
        if filtered.shape[0] < self.min_points:
            return points
        return filtered

    def _depth_to_meters(self, depth_image: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth_image)
        if np.issubdtype(depth.dtype, np.floating):
            return depth.astype(np.float64) * self.float_depth_scale
        return depth.astype(np.float64) * self.depth_scale

    def _normalize_camera_matrix(self, camera_matrix: Any) -> np.ndarray:
        if hasattr(camera_matrix, 'k'):
            camera_matrix = camera_matrix.k
        matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
            raise ValueError('Camera matrix fx and fy must be positive')
        return matrix

    def _validate_images(self, rgb_image: np.ndarray, depth_image: np.ndarray) -> None:
        if rgb_image.ndim not in (2, 3):
            raise ValueError(f'Unsupported rgb_image shape: {rgb_image.shape}')
        if depth_image.ndim != 2:
            raise ValueError(f'Depth image must be single-channel: {depth_image.shape}')
        if rgb_image.shape[:2] != depth_image.shape[:2]:
            raise ValueError(
                f'RGB image shape {rgb_image.shape[:2]} does not match depth '
                f'image shape {depth_image.shape[:2]}'
            )

    def _overlay_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        overlay = image.copy()
        overlay[mask.astype(bool)] = (0, 180, 255)
        return cv2.addWeighted(overlay, 0.35, image, 0.65, 0.0)

    def _draw_axis(
        self,
        image: np.ndarray,
        origin: np.ndarray,
        axis: np.ndarray,
        camera_matrix: np.ndarray,
        length: float,
        color: Tuple[int, int, int],
    ) -> None:
        self._draw_3d_segment(
            image,
            origin,
            origin + axis * length,
            camera_matrix,
            color,
            2,
        )

    def _draw_3d_segment(
        self,
        image: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
        camera_matrix: np.ndarray,
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        start_uv = self._project_point(start, camera_matrix)
        end_uv = self._project_point(end, camera_matrix)
        if start_uv is None or end_uv is None:
            return
        cv2.line(image, start_uv, end_uv, color, thickness, cv2.LINE_AA)
        cv2.circle(image, end_uv, 4, color, -1)

    def _project_point(
        self,
        point: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> Optional[Tuple[int, int]]:
        if point[2] <= 0.0 or not np.isfinite(point).all():
            return None
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        u = int(round(fx * point[0] / point[2] + cx))
        v = int(round(fy * point[1] / point[2] + cy))
        return (u, v)

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm < 1e-9:
            raise ValueError('Cannot normalize near-zero vector')
        return vector / norm

    def _rotation_matrix_to_quaternion(self, rotation: np.ndarray) -> np.ndarray:
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * scale
            qx = (rotation[2, 1] - rotation[1, 2]) / scale
            qy = (rotation[0, 2] - rotation[2, 0]) / scale
            qz = (rotation[1, 0] - rotation[0, 1]) / scale
        elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
            scale = np.sqrt(
                1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
            ) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / scale
            qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / scale
            qz = (rotation[0, 2] + rotation[2, 0]) / scale
        elif rotation[1, 1] > rotation[2, 2]:
            scale = np.sqrt(
                1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
            ) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / scale
            qx = (rotation[0, 1] + rotation[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = np.sqrt(
                1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
            ) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / scale
            qx = (rotation[0, 2] + rotation[2, 0]) / scale
            qy = (rotation[1, 2] + rotation[2, 1]) / scale
            qz = 0.25 * scale

        quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
        return quaternion / np.linalg.norm(quaternion)


def _load_image(path: Path, flags: int) -> np.ndarray:
    """Load an image from disk and fail with a clear message."""
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f'Failed to read image: {path}')
    return image


def _load_rgb_image(path: Path) -> np.ndarray:
    """Load a BGR color image for offline testing."""
    return _load_image(path, cv2.IMREAD_COLOR)


def _load_depth_image(path: Path) -> np.ndarray:
    """Load a depth image from ``.png`` or ``.npy`` files."""
    if path.suffix.lower() == '.npy':
        depth = np.load(path, allow_pickle=False)
    else:
        depth = _load_image(path, cv2.IMREAD_UNCHANGED)
    depth = np.asarray(depth)
    if depth.ndim != 2:
        raise ValueError(f'Depth image must be single-channel: {depth.shape}')
    return depth


def _load_mask_image(path: Path) -> np.ndarray:
    """Load a binary mask from ``.png`` or ``.npy`` files."""
    if path.suffix.lower() == '.npy':
        mask = np.load(path, allow_pickle=False)
    else:
        mask = _load_image(path, cv2.IMREAD_UNCHANGED)

    mask = np.asarray(mask)
    mask = np.squeeze(mask)
    if mask.ndim == 3:
        if mask.shape[2] == 4:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGRA2GRAY)
        else:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.ndim != 2:
        raise ValueError(f'Mask image must be single-channel: {mask.shape}')
    return mask


def _parse_camera_matrix(raw_value: str) -> np.ndarray:
    """Parse a flattened 3x3 camera matrix from a comma or space string."""
    values = [float(part) for part in str(raw_value).replace(',', ' ').split()]
    if len(values) != 9:
        raise ValueError('camera matrix must have 9 values')
    return np.asarray(values, dtype=np.float64).reshape(3, 3)


def _format_pose_summary(pose: GraspPose) -> str:
    """Return a compact multi-line summary for console output."""
    return '\n'.join([
        'Grasp pose:',
        f'  position={pose.position.tolist()}',
        f'  quaternion_xyzw={pose.quaternion_xyzw.tolist()}',
        f'  width={pose.width:.4f}m score={pose.score:.3f}',
        f'  center_pixel={pose.center_pixel} point_count={pose.point_count}',
    ])


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for offline grasp pose testing."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--rgb',
        type=Path,
        required=True,
        help='Path to RGB image.',
    )
    parser.add_argument(
        '--depth',
        type=Path,
        required=True,
        help='Path to aligned depth image (.png or .npy).',
    )
    parser.add_argument(
        '--mask',
        type=Path,
        required=True,
        help='Path to a binary object mask (.png or .npy).',
    )
    parser.add_argument(
        '--camera-matrix',
        type=str,
        required=True,
        help='Flattened 3x3 camera matrix, for example "fx,0,cx,0,fy,cy,0,0,1".',
    )
    parser.add_argument(
        '--depth-scale',
        type=float,
        default=0.001,
        help='Scale for integer depth images.',
    )
    parser.add_argument(
        '--float-depth-scale',
        type=float,
        default=1.0,
        help='Scale for floating-point depth images.',
    )
    parser.add_argument('--min-depth', type=float, default=0.05)
    parser.add_argument('--max-depth', type=float, default=3.0)
    parser.add_argument('--min-points', type=int, default=80)
    parser.add_argument('--max-points', type=int, default=20000)
    parser.add_argument('--outlier-percentile', type=float, default=95.0)
    parser.add_argument('--grasp-width-margin', type=float, default=0.02)
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('grasp_pose_overlay.png'),
        help='Path to save the visualization.',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the visualization window after inference.',
    )
    return parser.parse_args()


def main() -> None:
    """Run an offline grasp pose estimate from saved test images."""
    args = parse_args()
    rgb_image = _load_rgb_image(args.rgb)
    depth_image = _load_depth_image(args.depth)
    mask_image = _load_mask_image(args.mask)
    camera_matrix = _parse_camera_matrix(args.camera_matrix)

    estimator = GraspPoseEstimator(
        depth_scale=args.depth_scale,
        float_depth_scale=args.float_depth_scale,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        min_points=args.min_points,
        max_points=args.max_points,
        outlier_percentile=args.outlier_percentile,
        grasp_width_margin=args.grasp_width_margin,
    )
    pose = estimator.estimate(
        rgb_image,
        depth_image,
        camera_matrix,
        object_mask=mask_image,
    )
    overlay = estimator.visualize(
        rgb_image,
        pose,
        camera_matrix,
        object_mask=mask_image,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), overlay):
        raise RuntimeError(f'Failed to write overlay image to {args.output}')

    print(_format_pose_summary(pose))
    print(f'  overlay={args.output}')

    if args.show:
        cv2.imshow('grasp_pose_overlay', overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
