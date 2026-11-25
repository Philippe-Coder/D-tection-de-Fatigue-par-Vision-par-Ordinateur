"""Mesures géométriques extraites du visage (fonctions pures, sans caméra)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Indices du maillage MediaPipe (478 points).
RIGHT_EYE_EAR = (33, 160, 158, 133, 153, 144)
LEFT_EYE_EAR = (362, 385, 387, 263, 373, 380)
MOUTH = (78, 308, 13, 14)  # coin gauche, coin droit, haut, bas (lèvres internes)

RIGHT_EYE_LOOP = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)
LEFT_EYE_LOOP = (362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398)
LIPS_LOOP = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109,
)


@dataclass
class FaceMetrics:
    """Mesures instantanées d'un visage."""

    eye_closed: float      # 0 = grand ouvert, 1 = fermé
    mouth_open: float      # 0 = fermée, 1 = grande ouverte
    pitch: float           # degrés, positif = tête penchée vers le bas
    yaw: float             # degrés, rotation gauche/droite
    roll: float            # degrés, inclinaison sur l'épaule
    gaze_h: float = 0.0    # -1..1, regard horizontal relatif à la tête
    distance_cm: float = 0.0
    ear: float = 0.0       # eye aspect ratio géométrique (contrôle)
    ear3d: float = 0.0     # idem avec la profondeur, plus stable quand la tête bascule


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(pts: np.ndarray, idx: tuple[int, ...], dims: int = 2) -> float:
    """EAR de Soukupová & Čech : ~0.3 œil ouvert, < 0.15 œil fermé.

    Avec `dims=3`, la mesure utilise la profondeur des points : elle varie
    beaucoup moins quand la tête bascule en avant ou en arrière.
    """
    p1, p2, p3, p4, p5, p6 = (pts[i, :dims] for i in idx)
    horiz = _dist(p1, p4)
    if horiz < 1e-6:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * horiz)


def mouth_aspect_ratio(pts: np.ndarray) -> float:
    left, right, top, bottom = (pts[i, :2] for i in MOUTH)
    width = _dist(left, right)
    if width < 1e-6:
        return 0.0
    return _dist(top, bottom) / width


def euler_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    """(pitch, yaw, roll) en degrés depuis une matrice de rotation 3x3 ou 4x4.

    Repère MediaPipe : x à droite, y vers le haut, z vers la caméra.
    pitch > 0 : le haut de la tête bascule vers la caméra (tête baissée).
    """
    r = np.asarray(m, dtype=float)[:3, :3]
    sy = math.hypot(r[0, 0], r[1, 0])
    if sy > 1e-6:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    else:
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def metrics_from_face(
    landmarks: np.ndarray,
    blendshapes: dict[str, float] | None,
    matrix: np.ndarray | None,
    aspect: float = 1.0,
) -> FaceMetrics:
    """Combine points, blendshapes et pose en un jeu de mesures.

    `landmarks` : tableau (N, 3) normalisé ; `aspect` = largeur / hauteur de
    l'image, pour corriger les distances horizontales.
    """
    pts = landmarks.astype(float).copy()
    pts[:, 0] *= aspect
    if pts.shape[1] > 2:
        pts[:, 2] *= aspect  # z est exprimé dans la même unité que x
    ear = (eye_aspect_ratio(pts, LEFT_EYE_EAR) + eye_aspect_ratio(pts, RIGHT_EYE_EAR)) / 2.0
    ear3d = (eye_aspect_ratio(pts, LEFT_EYE_EAR, 3) + eye_aspect_ratio(pts, RIGHT_EYE_EAR, 3)) / 2.0
    mar = mouth_aspect_ratio(pts)

    if blendshapes:
        b = blendshapes
        eye_closed = (b.get("eyeBlinkLeft", 0.0) + b.get("eyeBlinkRight", 0.0)) / 2.0
        mouth_open = b.get("jawOpen", 0.0)
        gaze_left = b.get("eyeLookOutLeft", 0.0) - b.get("eyeLookInLeft", 0.0)
        gaze_right = b.get("eyeLookInRight", 0.0) - b.get("eyeLookOutRight", 0.0)
        gaze_h = (gaze_left + gaze_right) / 2.0
    else:
        # Repli purement géométrique.
        eye_closed = float(np.clip((0.30 - ear) / 0.20, 0.0, 1.0))
        mouth_open = float(np.clip(mar / 0.7, 0.0, 1.0))
        gaze_h = 0.0

    pitch = yaw = roll = 0.0
    distance = 0.0
    if matrix is not None:
        pitch, yaw, roll = euler_from_matrix(matrix)
        distance = abs(float(np.asarray(matrix)[2, 3]))

    return FaceMetrics(
        eye_closed=float(eye_closed),
        mouth_open=float(mouth_open),
        pitch=pitch,
        yaw=yaw,
        roll=roll,
        gaze_h=float(gaze_h),
        distance_cm=distance,
        ear=float(ear),
        ear3d=float(ear3d),
    )


EYE_CORNERS = ((33, 133), (362, 263))  # (coin, coin) de chaque œil


def eye_frames(landmarks: np.ndarray, width: int, height: int, margin: float = 0.45):
    """Cadre de chaque œil : (centre_x, centre_y, demi-côté, angle en degrés).

    L'angle est celui de la ligne des coins de l'œil : il permet de redresser
    le recadrage quand la tête est penchée sur l'épaule.
    """
    frames = []
    scale = np.array([width, height], dtype=float)
    for loop, (a, b) in zip((RIGHT_EYE_LOOP, LEFT_EYE_LOOP), EYE_CORNERS):
        pts = landmarks[list(loop), :2] * scale
        pa, pb = landmarks[a, :2] * scale, landmarks[b, :2] * scale
        d = pb - pa
        if d[0] < 0:
            d = -d
        half = float(np.hypot(*d)) / 2.0 * (1.0 + margin)
        angle = float(np.degrees(np.arctan2(d[1], d[0])))
        frames.append((float(pts[:, 0].mean()), float(pts[:, 1].mean()), half, angle))
    return frames


def eye_boxes(landmarks: np.ndarray, width: int, height: int, margin: float = 0.45):
    """Boîtes carrées (x0, y0, x1, y1) en pixels autour de chaque œil."""
    boxes = []
    for loop in (RIGHT_EYE_LOOP, LEFT_EYE_LOOP):
        xs = landmarks[list(loop), 0] * width
        ys = landmarks[list(loop), 1] * height
        cx, cy = float(xs.mean()), float(ys.mean())
        half = (float(xs.max() - xs.min()) / 2.0) * (1.0 + margin)
        x0, y0 = int(round(cx - half)), int(round(cy - half))
        x1, y1 = int(round(cx + half)), int(round(cy + half))
        boxes.append((max(0, x0), max(0, y0), min(width, x1), min(height, y1)))
    return boxes
