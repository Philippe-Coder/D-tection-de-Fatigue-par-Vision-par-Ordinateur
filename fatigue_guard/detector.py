"""Détection du visage : 478 points, blendshapes et pose (MediaPipe Face Landmarker)."""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import FACE_MODEL_URL, MODEL_DIR


@dataclass
class Face:
    landmarks: np.ndarray               # (478, 3) normalisé [0, 1]
    blendshapes: dict[str, float]
    matrix: np.ndarray | None           # transformation 4x4 (pose de la tête)

    @property
    def area(self) -> float:
        xs, ys = self.landmarks[:, 0], self.landmarks[:, 1]
        return float((xs.max() - xs.min()) * (ys.max() - ys.min()))


def ensure_face_model(progress=None) -> Path:
    """Télécharge le modèle de visage au premier lancement (~3,6 Mo)."""
    path = MODEL_DIR / "face_landmarker.task"
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if progress:
        progress("Téléchargement du modèle de visage…")
    tmp = path.with_suffix(".part")
    try:
        urllib.request.urlretrieve(FACE_MODEL_URL, tmp)
        tmp.replace(path)
    except Exception as exc:  # réseau indisponible, proxy…
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            "Impossible de télécharger le modèle de visage. Vérifiez la connexion "
            f"internet, ou placez le fichier manuellement dans {path}.\n({exc})"
        ) from exc
    return path


class FaceDetector:
    def __init__(self, max_faces: int = 2, progress=None):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        model = ensure_face_model(progress)
        options = vision.FaceLandmarkerOptions(
            # Passé en mémoire : évite les soucis de chemins accentués sous Windows.
            base_options=mp_python.BaseOptions(model_asset_buffer=model.read_bytes()),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame_rgb: np.ndarray, timestamp_ms: int) -> list[Face]:
        """Renvoie les visages, le plus grand (le plus proche) en premier."""
        ts = max(int(timestamp_ms), self._last_ts + 1)  # doit être strictement croissant
        self._last_ts = ts
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(frame_rgb))
        result = self._landmarker.detect_for_video(image, ts)

        faces: list[Face] = []
        for i, lms in enumerate(result.face_landmarks or []):
            pts = np.array([(p.x, p.y, p.z) for p in lms], dtype=np.float32)
            shapes: dict[str, float] = {}
            if result.face_blendshapes and i < len(result.face_blendshapes):
                shapes = {c.category_name: float(c.score) for c in result.face_blendshapes[i]}
            matrix = None
            mats = result.facial_transformation_matrixes
            if mats and i < len(mats):
                matrix = np.asarray(mats[i], dtype=np.float32)
            faces.append(Face(pts, shapes, matrix))
        faces.sort(key=lambda f: f.area, reverse=True)
        return faces

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass
