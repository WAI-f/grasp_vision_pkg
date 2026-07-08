# 代码逻辑与算法原理

## 1. 总体流程

当前 `grasp_vision_pkg` 的主流程是：

1. `camera_subscriber` 订阅 RGB、aligned depth、camera info。
2. 节点只缓存最新帧，不做实时推理。
3. client 调用 `/estimate_grasp_pose` service。
4. 节点默认启动时初始化一次 `SAM3OnnxSegmenter` 和 `GraspPoseEstimator`；如果关闭预加载，则首次 service 调用懒加载一次。
5. 后续 service 请求复用已加载并预热的 SAM3 ONNX sessions，将整张输入图 resize 到模型固定输入尺寸后执行分割推理，再把 mask/box 映射到原图坐标。
6. `GraspPoseEstimator` 用 mask + 深度图 + 相机内参估计抓取位姿。
7. service 返回 `success/status_code/message/pose/width/...`，client 自己决定下一步状态。

核心目标是把“高延迟、非确定成功”的推理动作从实时图像链路里剥离出来，改成按需请求式执行。

## 2. 模块分工

### 2.1 `camera_subscriber.py`

位置：[`grasp_vision_pkg/camera_subscriber.py`](../grasp_vision_pkg/camera_subscriber.py)

职责：

- 订阅图像和相机信息。
- 保存最近一次 RGB、depth、header、camera matrix。
- 暴露 ROS2 service：`/estimate_grasp_pose`。
- 启动时或首次请求时加载一次 SAM3/估计器，并在后续请求中复用。
- 把请求参数转换成 SAM3 prompt。
- 把 `GraspPoseEstimator` 的输出封装为 service response。
- 根据请求或参数选择是否发布 topic、保存 debug 图。

这个节点本身不负责算法细节，只负责调度和状态管理。

### 2.2 `sam3_onnx_segmenter.py`

位置：[`grasp_vision_pkg/sam3_onnx_segmenter.py`](../grasp_vision_pkg/sam3_onnx_segmenter.py)

职责：

- 加载三段式 SAM3 ONNX 模型：`image encoder`、`language encoder`、`decoder`。
- 将整张输入图按模型输入尺寸推理，并把预测 mask/box 映射到原图坐标。
- 可在初始化后执行一次空图预热，降低第一次 service 调用延迟。
- 把输入图像和 prompt 变成分割结果。
- 输出多个候选 mask、score、box，并选出一个最终对象。

它的输出是“对象分割”，不是抓取位姿。

### 2.3 `grasp_pose_estimator.py`

位置：[`grasp_vision_pkg/grasp_pose_estimator.py`](../grasp_vision_pkg/grasp_pose_estimator.py)

职责：

- 接收 object mask 和 aligned depth。
- 从 mask 内像素反投影出 3D 点。
- 对点云做 PCA，估计物体主轴和法向。
- 构造抓取位姿、夹爪宽度和质量分数。
- 输出 `GraspPose` 数据结构。

它负责“从对象区域推到抓取姿态”。

## 3. SAM3 分割原理

### 3.1 输入

`SAM3OnnxSegmenter.predict()` 的输入有两部分：

- 图像：BGR 或 RGB。
- prompt：
  - 文本 prompt，例如 `visual`、`cup`
  - normalized box prompt `[cx, cy, w, h]`
  - dict 格式 prompt，供 service 层或扩展调用

### 3.2 三段式 ONNX 链路

代码里实际是三步：

1. **Image encoder**
   - 将图像 resize 到固定输入尺寸。
   - 输出 vision position encoding 和 backbone FPN 特征。

2. **Language encoder**
   - 对 text prompt 做 tokenize。
   - 生成 language mask 和 language features。

3. **Decoder**
   - 把图像特征、语言特征、box prompt 一起送入 decoder。
   - 输出 boxes、scores、masks。

### 3.3 后处理

`_postprocess_decoder_output()` 做两件事：

- 将 box 从归一化坐标映射回像素坐标。
- 将 decoder mask resize 回原图大小，并用 `> 0.5` 二值化。

### 3.4 候选选择策略

`_select_index()` 不是选第一个 mask，而是：

- 先过滤 `score >= score_threshold` 的候选。
- 如果没有候选，再回退到全部候选。
- 在候选中取分数最高的那个。

这意味着 SAM3 模块输出的是“多候选 + 评分”，最终对象由打分规则选出。

## 4. 抓取位姿估计原理

### 4.1 从 mask 到 3D 点

`GraspPoseEstimator.estimate()` 的第一步是：

- 读取 mask。
- 在 aligned depth 图上，只保留 mask 内、深度有效、且在 `min_depth ~ max_depth` 范围内的像素。
- 用相机内参把每个像素反投影成 3D 点：

```text
x = (u - cx) * z / fx
y = (v - cy) * z / fy
z = depth
```

最后得到 `points` 和对应 `pixels`。

### 4.2 点云裁剪和去噪

算法先做两层过滤：

- 点数量必须至少达到 `min_points`。
- 如果点太多，先按均匀采样截到 `max_points`。
- 然后按点到中位数的距离做分位数剔除，去掉远离主体的异常点。

这部分的目的不是精细建模，而是让 PCA 受离群点影响更小。

### 4.3 相机平面 PCA 位姿估计

去噪后，只对 3D 点的相机 XY 分量做协方差矩阵和特征分解：

- 最大特征值对应物体在图像平面内的最长方向。
- 不再使用 3D PCA 的最小特征向量作为物体法向。

然后构造抓取坐标系：

- `long_axis`：物体在相机 XY 平面内的主长度方向，z 分量固定为 0。
- `camera_z_axis`：固定为相机坐标系 `[0, 0, 1]`，让抓取位姿 z 轴始终与相机 Z 轴平行。
- `closing_axis`：由 `long_axis × camera_z_axis` 得到，表示夹爪闭合轴。

最终旋转矩阵列向量是：

- x 轴：`closing_axis`
- y 轴：`long_axis`
- z 轴：`camera_z_axis`

### 4.4 位姿中心、宽度和分数

- `position`：点云中位数，作为对象中心。
- `width`：投影到闭合轴后的 5% 到 95% 分位跨度，再加上 `grasp_width_margin`。
- `score`：由两部分组成：
  - `anisotropy`，看点云是不是明显拉长
  - `density`，看有效点数量是否足够

这是一种启发式质量分数，不是学习式抓取成功率。

### 4.5 输出格式

`GraspPose` 包含：

- `position`
- `orientation_matrix`
- `quaternion_xyzw`
- `width`
- `score`
- `center_pixel`
- `point_count`

这让上层既能拿到 3D 姿态，也能拿到用于状态机判断的辅助值。

## 5. Service 层为什么这样设计

现在 service response 不只返回 `pose`，还返回：

- `status_code`
- `message`
- `point_count`
- `mask_pixel_count`
- `segmentation_score`
- `processing_time`

原因很直接：

- 推理不是实时的。
- 估计不一定成功。
- client 需要知道失败原因，而不是只看一个布尔值。

因此 service 层把“算法成功”和“运行状态”一起返回，方便 client 做状态机：

- 没有图像 -> 继续等
- prompt 不合法 -> 修正请求
- 模型不可用 -> 报依赖故障
- 分割或抓取失败 -> 重试或换视角

## 6. 当前实现的隐含假设

这个算法链路默认：

- RGB 和 depth 已经对齐。
- 相机内参是正确的。
- mask 对应的是单个主要对象。
- 对象形状有足够明显的主方向。
- 深度噪声不会大到破坏 PCA。

这些假设成立时效果比较稳；不成立时，返回的 `score`、`point_count` 和 `status_code` 会帮助上层判断是否重试。

## 7. 可改进点

后续如果要继续增强，建议优先看这几个方向：

- 允许 service 返回多个候选 grasp，而不是单个结果。
- 给 `GraspPoseEstimator` 加更严格的几何有效性检测。
- 给 `SAM3OnnxSegmenter` 加缓存，减少重复 prompt 的编码开销。
- 给 `camera_subscriber` 加超时和并发保护，避免重复服务请求重入。
- 把 `score` 从启发式指标升级成任务相关评分。

## 8. 建议的阅读顺序

如果你想快速看懂代码，建议按这个顺序读：

1. `camera_subscriber.py`
2. `sam3_onnx_segmenter.py`
3. `grasp_pose_estimator.py`
4. `test/test_camera_subscriber_service.py`
5. `test/test_grasp_pose_estimator.py`

这样能先看清数据流，再看算法本体。
