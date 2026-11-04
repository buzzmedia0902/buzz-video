#!/usr/bin/env python3
"""The Apple Vision helpers (faces / landmarks / lift / textboxes .swift) for machines without Apple Vision.

Same arguments, same JSON on stdout, so callers do not care which one ran (cv/vision.py picks):
  python vision.py faces     <folder>   {"f": name, "faces": [[x, y, w, h, conf, yaw], ...]}
  python vision.py landmarks <folder>   {"f": name, "faces": [{"box", "conf", "roll", "yaw", "pitch", "eye", "eye_l",
                                          "eye_r", "mouth_open", "mouth_w", "lift", "brow"}]}
  python vision.py textboxes <folder>   {"f": name, "lines": [{"text", "box": [x, y, w, h], "conf"}]}
  python vision.py lift in.png out_prefix [in2.png out_prefix2 ...]   -> out_prefix_person.png (8-bit person matte)
Boxes are normalised with origin top-left; angles in radians. MediaPipe for faces and mattes, RapidOCR for text.
Runs in its own venv (env.sh --vision); models download once into ~/.local/share/content-video/models.
"""
from __future__ import annotations

import json
import math
import os
import sys
import urllib.request

import numpy as np
from PIL import Image

MODELS = os.path.expanduser("~/.local/share/content-video/models")
_MP = "https://storage.googleapis.com/mediapipe-models"
URLS = {
    "detector": f"{_MP}/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    "landmarker": f"{_MP}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "segmenter": f"{_MP}/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite",
}
# face mesh indices (MediaPipe 478-point topology)
EYE_L = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)     # eye contours
EYE_R = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
LIPS_OUTER = (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146)
LIP_IN_TOP, LIP_IN_BOTTOM = 13, 14
BROWS = (70, 63, 105, 66, 107, 336, 296, 334, 293, 300)
CHIN = 152
# Apple Vision's face box is wider and starts higher than the mesh's brows..chin, and its eye / brow contours differ:
# these factors bring the numbers onto Vision's scale (fitted on the same 9 frames) so thumb.py's thresholds hold
BOX_W, BOX_H, BOX_UP = 1.2, 1.17, 0.11
EYE_SCALE, LIFT_SCALE, BROW_SCALE = 0.97, 0.72, 0.7


def model(key: str) -> str:
    path = os.path.join(MODELS, os.path.basename(URLS[key]))
    if not os.path.exists(path):
        os.makedirs(MODELS, exist_ok=True)
        print(f"[vision] downloading {os.path.basename(path)}", file=sys.stderr, flush=True)
        urllib.request.urlretrieve(URLS[key], path + ".part")
        os.replace(path + ".part", path)
    return path


def images(folder: str) -> list[str]:
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(".jpg"))


def mp_image(path: str):
    import mediapipe as mp
    rgb = np.ascontiguousarray(np.asarray(Image.open(path).convert("RGB")))
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), rgb.shape[1], rgb.shape[0]


def emit(row: dict) -> None:
    print(json.dumps(row, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------- faces
def faces(folder: str) -> None:
    from mediapipe.tasks.python import BaseOptions, vision
    det = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=model("detector")), min_detection_confidence=0.5))
    for name in images(folder):
        img, w, h = mp_image(os.path.join(folder, name))
        out = []
        for d in det.detect(img).detections:
            b = d.bounding_box
            kp = d.keypoints                      # right eye, left eye, nose tip, mouth, right ear, left ear
            yaw = 0.0
            if len(kp) >= 3:
                eye_dx = kp[1].x - kp[0].x
                if abs(eye_dx) > 1e-6:            # nose off the eyes' midpoint ~ head turned
                    yaw = max(-1.2, min(1.2, ((kp[2].x - (kp[0].x + kp[1].x) / 2) / eye_dx) * 1.6))
            out.append([b.origin_x / w, b.origin_y / h, b.width / w, b.height / h,
                        float(d.categories[0].score), yaw])
        emit({"f": name, "faces": out})


# ---------------------------------------------------------------- landmarks
def _euler(m: np.ndarray) -> tuple[float, float, float]:
    """(roll, yaw, pitch) from MediaPipe's face transformation matrix."""
    r = m[:3, :3]
    yaw = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    pitch = math.atan2(r[2, 1], r[2, 2])
    roll = math.atan2(r[1, 0], r[0, 0])
    return roll, yaw, pitch


def landmarks(folder: str) -> None:
    from mediapipe.tasks.python import BaseOptions, vision
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model("landmarker")), num_faces=4,
        output_facial_transformation_matrixes=True, min_face_detection_confidence=0.5))
    for name in images(folder):
        img, w, h = mp_image(os.path.join(folder, name))
        res = lm.detect(img)
        out = []
        for i, pts in enumerate(res.face_landmarks):
            p = np.array([[q.x * w, q.y * h] for q in pts])             # pixels, y down
            x0, x1 = p[:, 0].min(), p[:, 0].max()
            y0, y1 = p[list(BROWS), 1].min(), p[CHIN, 1]                 # brows..chin
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - BOX_UP * (y1 - y0)
            bw, bh = max((x1 - x0) * BOX_W, 1.0), max((y1 - y0) * BOX_H, 1.0)
            x0, y0 = cx - bw / 2, cy - bh / 2

            def ratio(e):                                                # contour height / width, like Vision
                c = p[list(e)]
                width = c[:, 0].max() - c[:, 0].min()
                return float((c[:, 1].max() - c[:, 1].min()) / width) * EYE_SCALE if width else 0.0
            eye_l, eye_r = ratio(EYE_L), ratio(EYE_R)
            outer = p[list(LIPS_OUTER)]
            corners_y = (p[61, 1] + p[291, 1]) / 2
            eye_y = p[list(EYE_L + EYE_R), 1].mean()
            roll = yaw = pitch = 0.0
            if res.facial_transformation_matrixes:
                roll, yaw, pitch = _euler(np.asarray(res.facial_transformation_matrixes[i]))
            out.append({
                "box": [x0 / w, y0 / h, bw / w, bh / h], "conf": 1.0,
                "roll": roll, "yaw": yaw, "pitch": pitch,
                "eye": (eye_l + eye_r) / 2, "eye_l": eye_l, "eye_r": eye_r,
                "mouth_open": float(abs(p[LIP_IN_BOTTOM, 1] - p[LIP_IN_TOP, 1]) / bh),
                "mouth_w": float((outer[:, 0].max() - outer[:, 0].min()) / bw),
                "lift": float((outer[:, 1].mean() - corners_y) / bh) * LIFT_SCALE,   # corners above centre: smile
                "brow": float((eye_y - p[list(BROWS), 1].mean()) / bh) * BROW_SCALE,
            })
        emit({"f": name, "faces": out})


# ---------------------------------------------------------------- lift
def lift(pairs: list[str]) -> None:
    from mediapipe.tasks.python import BaseOptions, vision
    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model("segmenter")), output_confidence_masks=True))
    for src, prefix in zip(pairs[::2], pairs[1::2]):
        img, w, h = mp_image(src)
        masks = seg.segment(img).confidence_masks           # 0 background, 1 hair, 2 body, 3 face, 4 clothes, 5 other
        person = 1.0 - np.squeeze(masks[0].numpy_view()).astype(np.float32)
        Image.fromarray((np.clip(person, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC) \
            .save(f"{prefix}_person.png")
        print(f"{prefix} person matte: {w} x {h}")


# ---------------------------------------------------------------- textboxes
def textboxes(folder: str) -> None:
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    for name in images(folder):
        path = os.path.join(folder, name)
        w, h = Image.open(path).size
        result, _ = ocr(path)
        lines = []
        for quad, text, conf in result or []:
            q = np.asarray(quad, np.float32)
            x0, y0 = q.min(axis=0)
            x1, y1 = q.max(axis=0)
            if (y1 - y0) / h < 0.02:                      # same floor as the Swift tool's minimumTextHeight
                continue
            lines.append({"text": text, "conf": float(conf),
                          "box": [float(x0 / w), float(y0 / h), float((x1 - x0) / w), float((y1 - y0) / h)]})
        emit({"f": name, "lines": lines})


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] not in ("faces", "landmarks", "textboxes", "lift"):
        raise SystemExit(__doc__)
    tool, args = sys.argv[1], sys.argv[2:]
    if tool == "lift":
        if len(args) % 2:
            raise SystemExit("usage: vision.py lift in.png out_prefix [in2.png out_prefix2 ...]")
        lift(args)
    else:
        {"faces": faces, "landmarks": landmarks, "textboxes": textboxes}[tool](args[0])


if __name__ == "__main__":
    main()
