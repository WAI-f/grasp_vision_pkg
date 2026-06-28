# grasp_vision_pkg

`grasp_vision_pkg` 是 RGB-D 抓取视觉包。节点订阅彩色图、对齐深度图和相机内参，缓存最新帧；抓取位姿估计通过 ROS2 service 按需触发，不在图像回调中实时推理。

接口定义统一放在 sibling package `robot_interface_pkg` 中，后续新增 `msg`、`srv`、`action` 也建议放到该包，方便其他 ROS2 package 复用。

## Package 结构

```text
/root/ros2_proj/moveit_example/
├── robot_interface_pkg/
│   └── srv/EstimateGraspPose.srv
└── grasp_vision_pkg/
    ├── config/config.yaml
    ├── launch/camera_subscriber.launch.py
    ├── models/sam3/
    │   ├── sam3_decoder.onnx
    │   ├── sam3_image_encoder.onnx
    │   ├── sam3_image_encoder.onnx.data
    │   ├── sam3_language_encoder.onnx
    │   └── sam3_language_encoder.onnx.data
    └── grasp_vision_pkg/
        ├── camera_subscriber.py
        ├── grasp_pose_estimator.py
        └── sam3_onnx_segmenter.py
```

## 代码逻辑和算法原理

详细说明见：[docs/code_logic_and_algorithm.md](docs/code_logic_and_algorithm.md)。

## 功能说明

- `camera_subscriber.py`
  - 订阅 RGB、aligned depth、camera info。
  - 缓存最新 RGB-D 帧和相机内参。
  - 提供 `/estimate_grasp_pose` service。
  - service 请求到达时才初始化 SAM3 ONNX segmenter 和 `GraspPoseEstimator`。
  - 可选发布抓取位姿、夹爪宽度、debug overlay 图像。

- `sam3_onnx_segmenter.py`
  - 使用 `models/sam3` 下的 SAM3 ONNX 模型做分割。
  - 支持 text prompt 和 normalized box prompt。
  - 可单独离线测试分割结果。

- `grasp_pose_estimator.py`
  - 根据物体 mask 和 aligned depth 点云估计抓取位姿。
  - 可单独使用已保存的 RGB、depth、mask 离线测试。

- `robot_interface_pkg`
  - 统一管理接口。
  - 当前包含 `robot_interface_pkg/srv/EstimateGraspPose`。

## 构建

从 workspace 根目录构建接口包和视觉包：

```bash
cd /root/ros2_proj/moveit_example
colcon build --packages-select robot_interface_pkg grasp_vision_pkg
source install/setup.bash
```

单独构建 `grasp_vision_pkg` 前，必须保证 `robot_interface_pkg` 已构建并 source 过。

## Service 接口

接口文件：

```text
robot_interface_pkg/srv/EstimateGraspPose
```

Python 引用：

```python
from robot_interface_pkg.srv import EstimateGraspPose
```

查看接口：

```bash
source /root/ros2_proj/moveit_example/install/setup.bash
ros2 interface show robot_interface_pkg/srv/EstimateGraspPose
```

请求字段：

| 字段 | 说明 |
| --- | --- |
| `use_default_prompt` | `true` 时使用节点参数中的默认 prompt。 |
| `prompt_type` | `text` 或 `box`。仅 `use_default_prompt=false` 时使用。 |
| `prompt` | text prompt，例如 `visual`、`cup`。 |
| `box_prompt` | normalized `[cx, cy, w, h]`，仅 box prompt 使用。 |
| `publish_result` | 本次请求是否发布 `/grasp/pose`、`/grasp/width`、`/grasp/debug_image`。 |
| `save_debug_image` | 本次请求是否保存抓取 debug overlay。 |

响应字段：

| 字段 | 说明 |
| --- | --- |
| `success` | 是否成功估计抓取位姿。 |
| `status_code` | 状态码，client 应优先根据它处理状态。 |
| `message` | 可读状态说明。 |
| `pose` | `geometry_msgs/PoseStamped` 抓取位姿。 |
| `width` | 估计夹爪开口宽度，单位 m。 |
| `score` | 抓取估计分数。 |
| `point_count` | 用于估计的有效点数量。 |
| `mask_pixel_count` | 选中 mask 的像素数量。 |
| `segmentation_score` | SAM3 选中 mask 的分割分数。 |
| `debug_image_path` | 保存的 debug overlay 路径。 |
| `processing_time` | 本次 service 处理耗时。 |

状态码：

| 状态码 | 含义 |
| --- | --- |
| `0 STATUS_SUCCESS` | 成功估计抓取位姿。 |
| `1 STATUS_NOT_READY` | 尚未收到 RGB、depth 或 camera info。 |
| `2 STATUS_INVALID_REQUEST` | 请求参数非法，例如 prompt 类型不支持。 |
| `3 STATUS_ESTIMATOR_UNAVAILABLE` | SAM3/估计器初始化失败。 |
| `4 STATUS_ESTIMATION_FAILED` | 已开始估计但未得到有效位姿。 |
| `5 STATUS_INTERNAL_ERROR` | 预留内部错误状态。 |

## 运行节点

使用 launch 文件：

```bash
cd /root/ros2_proj/moveit_example
source install/setup.bash
ros2 launch grasp_vision_pkg camera_subscriber.launch.py
```

指定配置文件：

```bash
ros2 launch grasp_vision_pkg camera_subscriber.launch.py \
  config_file:=/root/ros2_proj/moveit_example/grasp_vision_pkg/config/config.yaml
```

直接运行节点：

```bash
ros2 run grasp_vision_pkg camera_subscriber --ros-args \
  -p enable_grasp_pose:=true \
  -p save_debug_images:=false
```

## 调用抓取位姿估计

使用默认 prompt：

```bash
ros2 service call /estimate_grasp_pose robot_interface_pkg/srv/EstimateGraspPose \
"{use_default_prompt: true, prompt_type: text, prompt: visual, box_prompt: [0.0, 0.0, 0.0, 0.0], publish_result: false, save_debug_image: false}"
```

指定 text prompt：

```bash
ros2 service call /estimate_grasp_pose robot_interface_pkg/srv/EstimateGraspPose \
"{use_default_prompt: false, prompt_type: text, prompt: cup, box_prompt: [0.0, 0.0, 0.0, 0.0], publish_result: true, save_debug_image: true}"
```

指定 normalized box prompt：

```bash
ros2 service call /estimate_grasp_pose robot_interface_pkg/srv/EstimateGraspPose \
"{use_default_prompt: false, prompt_type: box, prompt: visual, box_prompt: [0.5, 0.5, 0.4, 0.4], publish_result: true, save_debug_image: true}"
```

无图像输入时，service 应返回：

```text
success=false
status_code=1
message='Waiting for color image, aligned depth image, and camera info.'
```

## 主要参数

配置文件：`config/config.yaml`

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `color_image_topic` | `/color/image_raw` | 彩色图 topic。 |
| `color_info_topic` | `/color/camera_info` | 彩色相机内参 topic。 |
| `aligned_depth_image_topic` | `/aligned_depth_to_color/image_raw` | 对齐到彩色相机的深度图 topic。 |
| `aligned_depth_info_topic` | `/aligned_depth_to_color/camera_info` | 对齐深度相机内参 topic。 |
| `enable_grasp_pose` | `true` | 是否启用抓取位姿 service。 |
| `grasp_pose_service` | `/estimate_grasp_pose` | service 名称。 |
| `publish_grasp_result` | `false` | 是否默认发布抓取结果 topic。 |
| `save_grasp_debug_image` | `true` | 是否默认保存抓取 debug overlay。 |
| `sam3_model_dir` | `''` | 空值时使用安装目录 `share/grasp_vision_pkg/models/sam3`。 |
| `sam3_prompt_type` | `text` | 默认 prompt 类型。 |
| `sam3_prompt` | `visual` | 默认 text prompt。 |
| `sam3_box_prompt` | `[0.0, 0.0, 0.0, 0.0]` | 默认 box prompt。 |
| `sam3_provider` | `CUDAExecutionProvider,CPUExecutionProvider` | ONNX Runtime provider 优先级。 |
| `min_depth` / `max_depth` | `0.05` / `3.0` | 有效深度范围，单位 m。 |
| `min_points` | `80` | 位姿估计所需最小有效点数量。 |
| `grasp_width_margin` | `0.02` | 夹爪宽度安全余量，单位 m。 |

## 输出 Topic

只有在参数 `publish_grasp_result=true` 或单次请求 `publish_result=true` 时发布。

| Topic | 类型 | 说明 |
| --- | --- | --- |
| `/grasp/pose` | `geometry_msgs/msg/PoseStamped` | 抓取位姿。 |
| `/grasp/width` | `std_msgs/msg/Float32` | 夹爪开口宽度，单位 m。 |
| `/grasp/debug_image` | `sensor_msgs/msg/Image` | 抓取 overlay 图像。 |

## 模型文件

SAM3 ONNX 模型放在：

```text
grasp_vision_pkg/models/sam3/
```

构建后会安装到：

```text
install/grasp_vision_pkg/share/grasp_vision_pkg/models/sam3/
```

当前需要以下文件：

```text
sam3_decoder.onnx
sam3_image_encoder.onnx
sam3_image_encoder.onnx.data
sam3_language_encoder.onnx
sam3_language_encoder.onnx.data
```
安装git-lfs, 并pull模型文件
```
git lfs pull
```

如果 `sam3_model_dir` 为空，节点会自动使用安装目录下的模型文件。源码环境或调试时也可以显式指定：

```bash
ros2 run grasp_vision_pkg camera_subscriber --ros-args \
  -p sam3_model_dir:=/root/ros2_proj/moveit_example/grasp_vision_pkg/models/sam3
```

## 离线测试

### 单独测试 GraspPoseEstimator

使用已保存的 RGB、depth、mask 和相机内参：

```bash
ros2 run grasp_vision_pkg grasp_pose_estimator \
  --rgb color_0_200000010.png \
  --depth aligned_depth_0_200000010.png \
  --mask mask.png \
  --camera-matrix "fx,0,cx,0,fy,cy,0,0,1" \
  --output grasp_pose_overlay.png
```

该测试不依赖 SAM3，只验证 mask 到抓取位姿估计链路。

### 单独测试 SAM3 ONNX 分割

```bash
ros2 run grasp_vision_pkg sam3_onnx_segmenter \
  --image color_0_200000010.png \
  --model-dir /root/ros2_proj/moveit_example/grasp_vision_pkg/models/sam3 \
  --text-prompt visual \
  --output sam3_overlay.png
```

box prompt 示例：

```bash
ros2 run grasp_vision_pkg sam3_onnx_segmenter \
  --image color_0_200000010.png \
  --model-dir /root/ros2_proj/moveit_example/grasp_vision_pkg/models/sam3 \
  --box-prompt 0.5,0.5,0.4,0.4 \
  --output sam3_overlay.png
```

## 测试和验证

从 workspace 根目录构建：

```bash
cd /root/ros2_proj/moveit_example
colcon build --packages-select robot_interface_pkg grasp_vision_pkg
source install/setup.bash
```

运行单元测试：

```bash
cd /root/ros2_proj/moveit_example/grasp_vision_pkg
pytest -q
```

检查 Python 语法：

```bash
python3 -m compileall grasp_vision_pkg test scripts
```

检查接口：

```bash
source /root/ros2_proj/moveit_example/install/setup.bash
ros2 interface show robot_interface_pkg/srv/EstimateGraspPose
python3 -c "from robot_interface_pkg.srv import EstimateGraspPose; print(EstimateGraspPose.__name__)"
```

检查模型是否安装：

```bash
find /root/ros2_proj/moveit_example/install/grasp_vision_pkg/share/grasp_vision_pkg/models/sam3 \
  -maxdepth 1 -type f -printf '%f\n' | sort
```

## Client 集成建议

client 不应只判断 `success`，建议优先处理 `status_code`：

- `STATUS_NOT_READY`：继续等待图像和相机内参。
- `STATUS_INVALID_REQUEST`：修正 prompt 或请求参数。
- `STATUS_ESTIMATOR_UNAVAILABLE`：检查 ONNX Runtime、SAM3 模型文件和依赖环境。
- `STATUS_ESTIMATION_FAILED`：可以换 prompt、换视角或重新采样。
- `STATUS_SUCCESS`：读取 `pose`、`width`、`score` 后进入抓取规划。

## 参考
1. sam3 onnx推理：[sam3-onnx](https://github.com/wkentaro/sam3-onnx)