"""Alertes sonores (sons générés localement, lecture non bloquante)."""
from __future__ import annotations

import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

from .config import DATA_DIR

SOUND_DIR = DATA_DIR / "sounds"
RATE = 22050


def _tone(freq: float, dur: float, volume: float) -> np.ndarray:
    t = np.linspace(0.0, dur, int(RATE * dur), endpoint=False)
    wave_ = np.sin(2 * np.pi * freq * t) + 0.3 * np.sin(4 * np.pi * freq * t)
    fade = max(1, int(RATE * 0.012))
    env = np.ones_like(t)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return volume * wave_ * env / 1.3


def _silence(dur: float) -> np.ndarray:
    return np.zeros(int(RATE * dur))


def _build(level: str) -> np.ndarray:
    if level == "alarm":   # sirène insistante
        beep = np.concatenate([_tone(1320, 0.14, 0.95), _silence(0.05),
                               _tone(990, 0.14, 0.95), _silence(0.05)])
        return np.tile(beep, 4)
    if level == "warn":
        return np.concatenate([_tone(880, 0.16, 0.6), _silence(0.07), _tone(660, 0.22, 0.6)])
    return np.concatenate([_tone(784, 0.12, 0.35), _tone(1047, 0.2, 0.35)])


def _sound_path(level: str) -> Path:
    path = SOUND_DIR / f"{level}.wav"
    if not path.exists():
        SOUND_DIR.mkdir(parents=True, exist_ok=True)
        pcm = (np.clip(_build(level), -1, 1) * 32767).astype("<i2")
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(pcm.tobytes())
    return path


def play(level: str) -> None:
    """Joue le son du niveau donné sans bloquer ; ne lève jamais d'exception."""
    try:
        path = str(_sound_path(level if level in ("alarm", "warn", "info") else "info"))
        if sys.platform == "win32":
            import winsound

            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
            return
        for player in ("afplay", "paplay", "aplay"):
            exe = shutil.which(player)
            if exe:
                subprocess.Popen([exe, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
        sys.stdout.write("\a")
        sys.stdout.flush()
    except Exception:
        pass
