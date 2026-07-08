from grasp_vision_pkg.sam3_onnx_segmenter import SAM3OnnxSegmenter
import numpy as np


def _segmenter_without_models():
    segmenter = SAM3OnnxSegmenter.__new__(SAM3OnnxSegmenter)
    segmenter.input_color_space = 'bgr'
    segmenter.input_size = (1008, 1008)
    return segmenter


def test_box_prompt_is_normalized_without_spatial_remapping():
    segmenter = _segmenter_without_models()

    box = segmenter._normalize_box_prompt([0.5, 0.5, 0.25, 0.25])

    assert np.allclose(box, [0.5, 0.5, 0.25, 0.25])


def test_postprocess_decoder_output_returns_original_image_shape():
    segmenter = _segmenter_without_models()
    boxes = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    masks = np.zeros((1, 1, 1008, 1008), dtype=np.float32)
    masks[0, 0, 100, 200] = 1.0

    restored_boxes, restored_masks = segmenter._postprocess_decoder_output(
        boxes,
        masks,
        image_width=1280,
        image_height=720,
    )

    assert restored_masks.shape == (1, 720, 1280)
    assert restored_masks.any()
    assert np.allclose(restored_boxes[0], [128.0, 144.0, 384.0, 288.0])
