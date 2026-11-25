"""CNN d'état des yeux (ouvert / fermé), optionnel.

Le modèle est entraîné par `fatigue-guard train` puis exporté en ONNX. S'il est
absent, l'application fonctionne avec les seules mesures MediaPipe. L'inférence
passe par OpenCV DNN : aucune dépendance lourde à l'exécution.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import MODEL_DIR, PROJECT_MODELS
from .features import eye_frames

INPUT_SIZE = 32
MODEL_NAME = "eye_state.onnx"


def find_model() -> Path | None:
    for folder in (PROJECT_MODELS, MODEL_DIR):
        path = folder / MODEL_NAME
        if path.exists():
            return path
    return None


def preprocess(gray: np.ndarray) -> np.ndarray:
    """Recadrage d'œil en niveaux de gris -> tenseur (1, 32, 32) dans [0, 1].

    Même traitement à l'entraînement et à l'inférence.
    """
    img = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    img = cv2.equalizeHist(img)
    return (img.astype(np.float32) / 255.0)[None, :, :]


def aligned_eye_crops(frame_bgr: np.ndarray, landmarks: np.ndarray, size: int = 64):
    """Recadrages d'yeux en niveaux de gris, redressés à l'horizontale.

    Le CNN a appris sur des yeux à peu près horizontaux : sans redressement,
    une tête penchée sur l'épaule lui présente des yeux en biais.
    """
    h, w = frame_bgr.shape[:2]
    crops = []
    for cx, cy, half, angle in eye_frames(landmarks, w, h):
        if half < 5:
            continue
        m = cv2.getRotationMatrix2D((cx, cy), angle, size / (2.0 * half))
        m[0, 2] += size / 2.0 - cx
        m[1, 2] += size / 2.0 - cy
        crop = cv2.warpAffine(frame_bgr, m, (size, size), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
        crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    return crops


class EyeStateCNN:
    def __init__(self, path: Path):
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.path = path

    @classmethod
    def load(cls) -> "EyeStateCNN | None":
        path = find_model()
        if path is None:
            return None
        try:
            return cls(path)
        except cv2.error:
            return None

    def closed_probability(self, frame_bgr: np.ndarray, landmarks: np.ndarray) -> float | None:
        """Probabilité moyenne (0..1) que les yeux soient fermés."""
        crops = [preprocess(c) for c in aligned_eye_crops(frame_bgr, landmarks)]
        if not crops:
            return None
        self.net.setInput(np.stack(crops))
        logits = self.net.forward().reshape(len(crops), -1)  # colonnes : [ouvert, fermé]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return float((e[:, 1] / e.sum(axis=1)).mean())
