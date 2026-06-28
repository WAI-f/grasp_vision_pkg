"""SAM3 ONNX segmentation backend and offline test CLI.

This module mirrors the three-stage ONNX flow from ``sam3-onnx/infer_onnx.py``:
image encoder -> language encoder -> decoder. It wraps that flow in a reusable
class that returns one selected object mask for grasping, plus a small CLI for
offline verification on saved images.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class SAM3Prediction:
    """Raw SAM3 outputs after postprocessing.

    ``boxes`` are in ``[x1, y1, x2, y2]`` pixel coordinates.
    """

    boxes: np.ndarray
    scores: np.ndarray
    masks: np.ndarray
    selected_index: int


class SAM3OnnxSegmenter:
    """SAM3 object segmenter backed by the exported ONNX models."""

    def __init__(
        self,
        model_dir: str | Path,
        providers: Sequence[str] | None = None,
        input_color_space: str = 'bgr',
        input_size: tuple[int, int] = (1008, 1008),
        score_threshold: float = 0.0,
    ):
        self.model_dir = Path(model_dir).expanduser()
        self.input_color_space = input_color_space.lower().strip()
        if self.input_color_space not in {'bgr', 'rgb'}:
            raise ValueError("input_color_space must be 'bgr' or 'rgb'")

        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.score_threshold = float(score_threshold)
        self.last_prediction: SAM3Prediction | None = None
        self.last_mask: np.ndarray | None = None

        self._onnxruntime = self._import_onnxruntime()
        self._tokenize = self._import_tokenize()
        self.providers = self._resolve_providers(providers)
        print(self.providers)

        self.image_encoder_path = self.model_dir / 'sam3_image_encoder.onnx'
        self.language_encoder_path = self.model_dir / 'sam3_language_encoder.onnx'
        self.decoder_path = self.model_dir / 'sam3_decoder.onnx'
        self._validate_model_files()

        self.sess_image = self._onnxruntime.InferenceSession(
            str(self.image_encoder_path),
            providers=self.providers,
        )
        self.sess_language = self._onnxruntime.InferenceSession(
            str(self.language_encoder_path),
            providers=self.providers,
        )
        self.sess_decode = self._onnxruntime.InferenceSession(
            str(self.decoder_path),
            providers=self.providers,
        )

    def predict(self, image: np.ndarray, prompt: Any = None) -> SAM3Prediction:
        """Return SAM3 masks, scores and boxes for a prompt.

        ``image`` may be an OpenCV BGR image (default) or RGB if
        ``input_color_space='rgb'`` was selected in the constructor.
        ``prompt`` can be a text string, a normalized box ``[cx, cy, w, h]``,
        or a dict like ``{'type': 'text', 'value': 'cup'}``.
        """
        bgr_image = self._normalize_display_image(image)
        text_prompt, box_prompt = self._normalize_prompt(prompt)

        vision_pos_enc, backbone_fpn = self._encode_image(bgr_image)
        language_mask, language_features = self._encode_language(text_prompt)
        boxes, scores, masks = self._decode(
            backbone_fpn=backbone_fpn,
            vision_pos_enc_2=vision_pos_enc[2],
            language_mask=language_mask,
            language_features=language_features,
            box_prompt=box_prompt,
        )
        boxes, masks = self._postprocess_decoder_output(
            boxes=boxes,
            masks=masks,
            image_width=bgr_image.shape[1],
            image_height=bgr_image.shape[0],
        )

        scores = np.asarray(scores).reshape(-1)
        if boxes.shape[0] != masks.shape[0] or scores.shape[0] != masks.shape[0]:
            raise ValueError(
                'SAM3 decoder returned inconsistent boxes, scores, and masks '
                f'({boxes.shape}, {scores.shape}, {masks.shape})'
            )
        if masks.shape[0] == 0:
            raise ValueError('SAM3 decoder returned no masks')

        selected_index = self._select_index(scores)
        prediction = SAM3Prediction(
            boxes=boxes,
            scores=scores,
            masks=masks,
            selected_index=selected_index,
        )
        self.last_prediction = prediction
        self.last_mask = prediction.masks[prediction.selected_index]
        return prediction

    def segment(self, image: np.ndarray, prompt: Any = None) -> np.ndarray:
        """Return the selected object mask as a boolean array."""
        prediction = self.predict(image=image, prompt=prompt)
        return prediction.masks[prediction.selected_index]

    def visualize(
        self,
        image: np.ndarray,
        prediction: SAM3Prediction,
        prompt_label: str | None = None,
    ) -> np.ndarray:
        """Overlay masks and boxes on the input image."""
        vis = self._normalize_display_image(image).copy()
        if prediction.masks.shape[0] == 0:
            return vis

        palette = [
            (0, 165, 255),
            (0, 255, 0),
            (255, 0, 0),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
            (128, 0, 255),
            (0, 128, 255),
        ]

        for idx, mask in enumerate(prediction.masks):
            color = palette[idx % len(palette)]
            alpha = 0.45 if idx == prediction.selected_index else 0.25
            vis = self._blend_mask(vis, mask, color, alpha=alpha)

            x1, y1, x2, y2 = self._clip_box(prediction.boxes[idx], vis.shape[1], vis.shape[0])
            thickness = 3 if idx == prediction.selected_index else 2
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

            caption = f'{idx}: {prediction.scores[idx]:.2f}'
            if idx == prediction.selected_index:
                caption = f'selected {caption}'
            self._draw_caption(vis, caption, (x1, max(0, y1 - 8)), color)

        if prompt_label:
            self._draw_caption(vis, f'prompt: {prompt_label}', (12, 26), (255, 255, 255))

        return vis

    def _validate_model_files(self) -> None:
        missing = [str(path) for path in [
            self.image_encoder_path,
            self.language_encoder_path,
            self.decoder_path,
        ] if not path.exists()]
        if missing:
            raise FileNotFoundError(
                'SAM3 ONNX model files are missing: ' + ', '.join(missing)
            )

    def _resolve_providers(self, providers: Sequence[str] | None) -> list[str]:
        available = list(self._onnxruntime.get_available_providers())
        print(available)
        if providers is None:
            preferred = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            preferred = [str(provider).strip() for provider in providers if str(provider).strip()]
            if 'CPUExecutionProvider' not in preferred:
                preferred.append('CPUExecutionProvider')

        resolved = [provider for provider in preferred if provider in available]
        if not resolved:
            if 'CPUExecutionProvider' in available:
                resolved = ['CPUExecutionProvider']
            elif available:
                resolved = [available[0]]
            else:
                raise RuntimeError('No ONNX Runtime execution providers are available')
        return resolved

    def _import_onnxruntime(self):
        try:
            import onnxruntime
        except ImportError as exc:  # pragma: no cover - runtime environment issue
            raise RuntimeError(
                'onnxruntime is required for SAM3 ONNX inference. '
                'Activate the sam3-onnx environment or install onnxruntime.'
            ) from exc
        return onnxruntime

    def _import_tokenize(self):
        try:
            from osam._models.yoloworld.clip import tokenize
        except ImportError as exc:  # pragma: no cover - runtime environment issue
            raise RuntimeError(
                'osam is required for SAM3 text prompts. '
                'Activate the sam3-onnx environment or install osam.'
            ) from exc
        return tokenize

    def _normalize_prompt(self, prompt: Any) -> tuple[str, list[float] | None]:
        if prompt is None:
            return 'visual', None

        if isinstance(prompt, str):
            text_prompt = prompt.strip() or 'visual'
            return text_prompt, None

        if isinstance(prompt, dict):
            prompt_type = str(prompt.get('type', 'text')).lower()
            if prompt_type == 'text':
                text_prompt = str(
                    prompt.get('value', prompt.get('text_prompt', 'visual'))
                ).strip() or 'visual'
                box_prompt = prompt.get('box_prompt')
                if box_prompt is not None:
                    return text_prompt, self._normalize_box_prompt(box_prompt)
                return text_prompt, None

            if prompt_type == 'box':
                text_prompt = str(prompt.get('text_prompt', 'visual')).strip() or 'visual'
                box_prompt = prompt.get('value', prompt.get('box_prompt'))
                return text_prompt, self._normalize_box_prompt(box_prompt)

            raise ValueError("prompt['type'] must be 'text' or 'box'")

        if self._looks_like_box(prompt):
            return 'visual', self._normalize_box_prompt(prompt)

        return str(prompt), None

    def _looks_like_box(self, value: Any) -> bool:
        if isinstance(value, np.ndarray):
            return value.size == 4
        if isinstance(value, (list, tuple)):
            return len(value) == 4
        return False

    def _normalize_box_prompt(self, box_prompt: Any) -> list[float]:
        if not self._looks_like_box(box_prompt):
            raise ValueError('box prompt must have 4 values: cx, cy, w, h')
        box = [float(x) for x in np.asarray(box_prompt).reshape(-1).tolist()]
        if len(box) != 4:
            raise ValueError('box prompt must have 4 values: cx, cy, w, h')
        return box

    def _normalize_display_image(self, image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim == 2:
            return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
        if array.ndim != 3:
            raise ValueError(f'Unsupported image shape: {array.shape}')

        if array.shape[2] == 4:
            if self.input_color_space == 'rgb':
                return cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
            return cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
        if array.shape[2] != 3:
            raise ValueError(f'Unsupported image shape: {array.shape}')

        if self.input_color_space == 'rgb':
            return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        return array.copy()

    def _encode_image(self, bgr_image: np.ndarray):
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_image, self.input_size, interpolation=cv2.INTER_LINEAR)
        image_tensor = np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.uint8)
        output = self.sess_image.run(None, {'image': image_tensor})
        if len(output) != 6:
            raise ValueError(f'image encoder returned {len(output)} outputs, expected 6')
        vision_pos_enc = list(output[:3])
        backbone_fpn = list(output[3:])
        return vision_pos_enc, backbone_fpn

    def _encode_language(self, text_prompt: str):
        tokens = self._tokenize(texts=[text_prompt], context_length=32)
        output = self.sess_language.run(None, {'tokens': tokens})
        if len(output) != 3:
            raise ValueError(
                f'language encoder returned {len(output)} outputs, expected 3'
            )
        language_mask = output[0]
        language_features = output[1]
        return language_mask, language_features

    def _decode(
        self,
        backbone_fpn: Sequence[np.ndarray],
        vision_pos_enc_2: np.ndarray,
        language_mask: np.ndarray,
        language_features: np.ndarray,
        box_prompt: list[float] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        box_coords = np.array(
            box_prompt if box_prompt is not None else [0, 0, 0, 0],
            dtype=np.float32,
        ).reshape(1, 1, 4)
        box_labels = np.array([[1]], dtype=np.int64)
        box_masks = np.array(
            [False] if box_prompt is not None else [True],
            dtype=np.bool_,
        ).reshape(1, 1)

        output = self.sess_decode.run(
            None,
            {
                'backbone_fpn_0': backbone_fpn[0],
                'backbone_fpn_1': backbone_fpn[1],
                'backbone_fpn_2': backbone_fpn[2],
                'vision_pos_enc_2': vision_pos_enc_2,
                'language_mask': language_mask,
                'language_features': language_features,
                'box_coords': box_coords,
                'box_labels': box_labels,
                'box_masks': box_masks,
            },
        )
        if len(output) != 3:
            raise ValueError(f'decoder returned {len(output)} outputs, expected 3')
        boxes = np.asarray(output[0])
        scores = np.asarray(output[1])
        masks = np.asarray(output[2])
        return boxes, scores, masks

    def _postprocess_decoder_output(
        self,
        boxes: np.ndarray,
        masks: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        boxes = boxes * np.array(
            [image_width, image_height, image_width, image_height],
            dtype=np.float32,
        )
        if len(masks) == 0:
            return boxes, np.empty((0, image_height, image_width), dtype=bool)
        masks = np.array(
            [
                cv2.resize(
                    mask[0],
                    dsize=(image_width, image_height),
                    interpolation=cv2.INTER_LINEAR,
                ) > 0.5
                for mask in masks
            ]
        )
        return boxes, masks

    def _select_index(self, scores: np.ndarray) -> int:
        if scores.size == 0:
            raise ValueError('SAM3 decoder returned no scores')
        scores = np.asarray(scores).reshape(-1)
        valid = np.flatnonzero(scores >= self.score_threshold)
        if valid.size == 0:
            valid = np.arange(scores.size)
        return int(valid[np.argmax(scores[valid])])

    def _blend_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color: tuple[int, int, int],
        alpha: float,
    ) -> np.ndarray:
        blended = image.copy()
        mask = np.asarray(mask).astype(bool)
        if not mask.any():
            return blended
        overlay = np.zeros_like(blended)
        overlay[:] = color
        blended = np.where(mask[:, :, None], overlay, blended)
        return cv2.addWeighted(blended, alpha, image, 1.0 - alpha, 0.0)

    def _clip_box(self, box: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = [int(round(v)) for v in np.asarray(box).reshape(-1)[:4]]
        x1 = int(np.clip(x1, 0, max(0, width - 1)))
        x2 = int(np.clip(x2, 0, max(0, width - 1)))
        y1 = int(np.clip(y1, 0, max(0, height - 1)))
        y2 = int(np.clip(y2, 0, max(0, height - 1)))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return x1, y1, x2, y2

    def _draw_caption(
        self,
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        x, y = origin
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        top_left = (x, max(0, y - th - baseline - 4))
        bottom_right = (x + tw + 6, max(0, y + 2))
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)
        cv2.putText(
            image,
            text,
            (x + 3, max(0, y - 3)),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def _parse_box_prompt(raw_value: str | None, image: np.ndarray) -> list[float]:
    if raw_value is None:
        raise ValueError('box prompt is required')
    values = [float(x) for x in str(raw_value).split(',')]
    if len(values) != 4:
        raise ValueError('box prompt must have 4 values: cx, cy, w, h')
    if values == [0.0, 0.0, 0.0, 0.0]:
        x, y, w, h = cv2.selectROI(
            'Select box prompt and press ENTER or SPACE',
            image,
            fromCenter=False,
            showCrosshair=True,
        )
        cv2.destroyAllWindows()
        if [x, y, w, h] == [0, 0, 0, 0]:
            raise ValueError('No box prompt selected')
        return [
            (x + w / 2) / image.shape[1],
            (y + h / 2) / image.shape[0],
            w / image.shape[1],
            h / image.shape[0],
        ]
    return values


def _make_prompt_label(prompt: Any) -> str:
    if prompt is None:
        return 'visual'
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, dict):
        prompt_type = str(prompt.get('type', 'text')).lower()
        if prompt_type == 'text':
            return str(prompt.get('value', prompt.get('text_prompt', 'visual')))
        if prompt_type == 'box':
            return str(prompt.get('text_prompt', 'visual'))
    return 'visual'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--image',
        type=Path,
        required=True,
        help='Path to the input image.',
    )
    parser.add_argument(
        '--model-dir',
        type=Path,
        required=True,
        help='Directory containing the three exported SAM3 ONNX models.',
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        '--text-prompt',
        type=str,
        help='Text prompt for segmentation.',
    )
    prompt_group.add_argument(
        '--box-prompt',
        type=str,
        nargs='?',
        const='0,0,0,0',
        help='Box prompt in normalized cx,cy,w,h format. A blank value opens ROI selection.',
    )
    parser.add_argument(
        '--provider',
        type=str,
        default='CPUExecutionProvider',
        help='Comma-separated ONNX Runtime execution providers in priority order.',
    )
    parser.add_argument(
        '--score-threshold',
        type=float,
        default=0.0,
        help='Minimum score to prefer when selecting the final mask.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('sam3_overlay.png'),
        help='Path to save the visualized prediction.',
    )
    parser.add_argument(
        '--mask-output',
        type=Path,
        default=None,
        help='Optional path to save the selected mask as a binary PNG.',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show the visualization window after inference.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Failed to read image: {args.image}')

    providers = [provider.strip() for provider in args.provider.split(',') if provider.strip()]
    segmenter = SAM3OnnxSegmenter(
        model_dir=args.model_dir,
        providers=providers,
        input_color_space='bgr',
        score_threshold=args.score_threshold,
    )

    if args.text_prompt is not None:
        prompt: Any = args.text_prompt
    else:
        box_prompt = _parse_box_prompt(args.box_prompt, image)
        prompt = {'type': 'box', 'value': box_prompt, 'text_prompt': 'visual'}

    prediction = segmenter.predict(image, prompt=prompt)
    overlay = segmenter.visualize(image, prediction, prompt_label=_make_prompt_label(prompt))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), overlay):
        raise RuntimeError(f'Failed to write overlay image to {args.output}')

    if args.mask_output is not None:
        args.mask_output.parent.mkdir(parents=True, exist_ok=True)
        mask = prediction.masks[prediction.selected_index].astype(np.uint8) * 255
        if not cv2.imwrite(str(args.mask_output), mask):
            raise RuntimeError(f'Failed to write mask image to {args.mask_output}')

    selected_box = prediction.boxes[prediction.selected_index].tolist()
    selected_score = float(prediction.scores[prediction.selected_index])
    print(
        'SAM3 prediction:',
        f'masks={prediction.masks.shape}',
        f'selected_index={prediction.selected_index}',
        f'score={selected_score:.4f}',
        f'box={selected_box}',
        f'overlay={args.output}',
        sep='\n  ',
    )
    if args.mask_output is not None:
        print(f'  mask={args.mask_output}')

    if args.show:
        cv2.imshow('sam3_overlay', overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
