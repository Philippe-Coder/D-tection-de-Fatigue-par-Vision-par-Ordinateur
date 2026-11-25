"""Profils d'utilisation et constantes globales."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

APP_NAME = "Fatigue Guard"
DATA_DIR = Path.home() / ".fatigue_guard"
MODEL_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "history.db"
PROJECT_MODELS = Path(__file__).resolve().parent.parent / "models"

DEFAULT_PORT = 8765

FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


@dataclass(frozen=True)
class Profile:
    """Réglages d'un contexte d'utilisation.

    Les durées sont en secondes, les angles en degrés. Un niveau vaut
    "info", "warn" ou "alarm" ; une durée à 0 désactive la règle.
    """

    key: str
    label: str
    description: str
    microsleep_s: float = 1.5      # yeux fermés en continu -> alarme
    look_away_s: float = 8.0       # tête / regard détourné -> alerte
    look_away_level: str = "info"
    head_down_s: float = 4.0       # tête baissée (chute, téléphone)
    head_down_level: str = "warn"
    posture_s: float = 5.0         # tête en arrière / sur l'épaule (posture de sommeil)
    absent_s: float = 10.0         # visage absent
    absent_level: str = "info"
    multi_face: bool = False       # alerte si plusieurs visages
    break_min: float = 45.0        # rappel de pause (0 = désactivé)
    eye_strain: bool = True        # conseils fatigue visuelle
    yaw_deg: float = 28.0          # tolérance de rotation de la tête
    pitch_deg: float = 20.0        # tolérance d'inclinaison vers le bas


PROFILES: dict[str, Profile] = {
    p.key: p
    for p in [
        Profile(
            key="bureau",
            label="Travail au bureau",
            description="Travail sur PC : somnolence, pauses régulières, tolérant "
            "sur les regards hors écran.",
        ),
        Profile(
            key="conduite",
            label="Conduite",
            description="Réactions très rapides : micro-sommeil, regard hors de la "
            "route, tête qui tombe.",
            microsleep_s=0.8,
            look_away_s=2.0,
            look_away_level="alarm",
            head_down_s=1.5,
            head_down_level="alarm",
            posture_s=1.5,
            absent_s=3.0,
            absent_level="warn",
            break_min=120.0,
            eye_strain=False,
            yaw_deg=32.0,
            pitch_deg=18.0,
        ),
        Profile(
            key="ecran",
            label="Confort visuel",
            description="Longues sessions d'écran : clignements, distance à "
            "l'écran, règle 20-20-20.",
            microsleep_s=2.0,
            look_away_s=0.0,
            head_down_s=0.0,
            absent_s=0.0,
            break_min=20.0,
        ),
        Profile(
            key="etude",
            label="Étude / concentration",
            description="Apprentissage : signale quand l'attention quitte le "
            "support (regard ailleurs, téléphone, absence).",
            look_away_s=5.0,
            look_away_level="warn",
            head_down_s=6.0,
            absent_s=8.0,
            absent_level="warn",
            break_min=30.0,
        ),
        Profile(
            key="examen",
            label="Examen",
            description="Surveillance stricte : regard hors écran, absence, "
            "présence d'une autre personne.",
            look_away_s=3.0,
            look_away_level="warn",
            head_down_s=4.0,
            absent_s=4.0,
            absent_level="alarm",
            multi_face=True,
            break_min=0.0,
            eye_strain=False,
            yaw_deg=24.0,
        ),
    ]
}

DEFAULT_PROFILE = "bureau"


def get_profile(key: str | None) -> Profile:
    return PROFILES.get(key or DEFAULT_PROFILE, PROFILES[DEFAULT_PROFILE])
