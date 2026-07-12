from grasp_vision_pkg.sam3_onnx_segmenter import SAM3OnnxSegmenter, _CropMeta
import numpy as np


def _segmenter_without_models():
    segmenter = SAM3OnnxSegmenter.__new__(SAM3OnnxSegmenter)
    segmenter.input_color_space = 'bgr'
    segmenter.input_size = (672, 672)
    segmenter.crop_mode = 'center_square'
    return segmenter


def test_box_prompt_is_normalized_without_spatial_remapping():
    segmenter = _segmenter_without_models()

    box = segmenter._normalize_box_prompt([0.5, 0.5, 0.25, 0.25])

    assert np.allclose(box, [0.5, 0.5, 0.25, 0.25])


def test_make_crop_meta_uses_center_square_crop():
    segmenter = _segmenter_without_models()
    image = np.zeros((720, 1280, 3), dtype=np.uint8)

    crop_meta = segmenter._make_crop_meta(image)

    assert crop_meta == _CropMeta(
        original_width=1280,
        original_height=720,
        crop_x=280,
        crop_y=0,
        crop_width=720,
        crop_height=720,
    )


def test_box_prompt_is_remapped_to_crop_coordinates():
    segmenter = _segmenter_without_models()
    crop_meta = _CropMeta(
        original_width=1280,
        original_height=720,
        crop_x=280,
        crop_y=0,
        crop_width=720,
        crop_height=720,
    )

    box = segmenter._remap_box_prompt_to_crop([0.5, 0.5, 0.25, 0.5], crop_meta)

    assert np.allclose(box, [0.5, 0.5, 4.0 / 9.0, 0.5])


def test_postprocess_decoder_output_returns_original_image_shape():
    segmenter = _segmenter_without_models()
    boxes = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    masks = np.zeros((1, 1, 672, 672), dtype=np.float32)
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


def test_postprocess_decoder_output_applies_crop_offset():
    segmenter = _segmenter_without_models()
    boxes = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    masks = np.zeros((1, 1, 672, 672), dtype=np.float32)
    masks[0, 0, 100, 200] = 1.0
    crop_meta = _CropMeta(
        original_width=1280,
        original_height=720,
        crop_x=280,
        crop_y=0,
        crop_width=720,
        crop_height=720,
    )

    restored_boxes, restored_masks = segmenter._postprocess_decoder_output(
        boxes,
        masks,
        image_width=1280,
        image_height=720,
        crop_meta=crop_meta,
    )

    assert restored_masks.shape == (1, 720, 1280)
    assert restored_masks.any()
    assert not restored_masks[0, :, :280].any()
    assert not restored_masks[0, :, 1000:].any()
    assert np.allclose(restored_boxes[0], [352.0, 144.0, 496.0, 288.0])
