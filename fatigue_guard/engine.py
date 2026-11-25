"""Moteur : caméra -> visage -> mesures -> analyse, dans un thread dédié."""
from __future__ import annotations

import sys
import threading
import time
from collections import deque
from typing import Callable

import cv2
import numpy as np

from . import alerts as sound
from .analyzer import FatigueAnalyzer, Snapshot
from .config import Profile, get_profile
from .detector import Face, FaceDetector
from .eye_cnn import EyeStateCNN
from .features import metrics_from_face
from .storage import SessionRecorder

UpdateCallback = Callable[[np.ndarray, Snapshot, "Face | None"], None]


def open_camera(index: int) -> cv2.VideoCapture | None:
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform == "win32" else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    return None


class Engine:
    """Boucle de capture et d'analyse.

    `on_update(frame_bgr, snapshot, face)` est appelé depuis le thread du
    moteur à chaque image ; `on_error(message)` en cas de panne ;
    `on_status(message)` pour les étapes de démarrage.
    """

    def __init__(self, profile: str | Profile = "bureau", camera: int = 0, *,
                 sound_enabled: bool = True, record: bool = True, use_cnn: bool = True,
                 on_update: UpdateCallback | None = None,
                 on_error: Callable[[str], None] | None = None,
                 on_status: Callable[[str], None] | None = None):
        self.profile = profile if isinstance(profile, Profile) else get_profile(profile)
        self.camera = camera
        self.sound_enabled = sound_enabled
        self.record = record
        self.use_cnn = use_cnn
        self.on_update = on_update
        self.on_error = on_error
        self.on_status = on_status

        self.latest = Snapshot(profile=self.profile.key)
        self.last_signals: dict = {}   # signaux bruts de la dernière image (diagnostic)
        self.analyzer: FatigueAnalyzer | None = None
        self.fps = 0.0
        self.cnn_active = False
        self.running = False
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._recalibrate = threading.Event()
        self._new_profile: Profile | None = None
        self._lock = threading.Lock()
        self._recent_alerts: deque[dict] = deque(maxlen=20)
        self._alert_seq = 0

    # -------------------------------------------------------------- contrôle
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="fatigue-engine", daemon=True)
        self._thread.start()

    def stop(self, wait: bool = True) -> None:
        self._stop.set()
        if wait and self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)

    def set_profile(self, key: str) -> None:
        with self._lock:
            self._new_profile = get_profile(key)

    def recalibrate(self) -> None:
        self._recalibrate.set()

    def status(self) -> dict:
        """État pour le serveur local."""
        data = self.latest.to_dict()
        del data["alerts"]  # propres à une image ; les extensions lisent recent_alerts
        data["recent_alerts"] = list(self._recent_alerts)
        data["running"] = self.running
        data["fps"] = round(self.fps, 1)
        data["profile_label"] = self.profile.label
        for key, value in list(data.items()):
            if isinstance(value, float):
                data[key] = round(value, 2)
        return data

    # ---------------------------------------------------------------- boucle
    def _say(self, msg: str) -> None:
        if self.on_status:
            self.on_status(msg)

    def _run(self) -> None:
        cap = None
        detector = None
        recorder = None
        analyzer = None
        t_start = time.monotonic()
        try:
            self._say("Chargement du modèle de visage…")
            detector = FaceDetector(max_faces=2, progress=self._say)
            cnn = EyeStateCNN.load() if self.use_cnn else None
            self.cnn_active = cnn is not None

            self._say("Ouverture de la caméra…")
            cap = open_camera(self.camera)
            if cap is None:
                raise RuntimeError(
                    f"Caméra {self.camera} inaccessible. Vérifiez qu'aucune autre application "
                    "ne l'utilise et que l'accès est autorisé dans Paramètres > Confidentialité."
                )

            analyzer = FatigueAnalyzer(self.profile)
            if self.record:
                recorder = SessionRecorder(self.profile.key)
            self.running = True
            self._say("")
            t_start = time.monotonic()
            fails = 0
            last = t_start

            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    fails += 1
                    if fails > 30:
                        raise RuntimeError("Le flux de la caméra s'est interrompu.")
                    time.sleep(0.03)
                    continue
                fails = 0
                frame = cv2.flip(frame, 1)  # effet miroir, plus naturel
                now = time.monotonic()

                with self._lock:
                    if self._new_profile is not None:
                        # Nouveau profil = nouvelle session.
                        if recorder:
                            recorder.close(analyzer, now - t_start)
                        self.profile = self._new_profile
                        self._new_profile = None
                        analyzer = FatigueAnalyzer(self.profile)
                        recorder = SessionRecorder(self.profile.key) if self.record else None
                        t_start = now
                if self._recalibrate.is_set():
                    self._recalibrate.clear()
                    analyzer.recalibrate()

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = detector.detect(rgb, int((now - t_start) * 1000))
                face = faces[0] if faces else None
                metrics = None
                if face is not None:
                    h, w = frame.shape[:2]
                    metrics = metrics_from_face(face.landmarks, face.blendshapes, face.matrix, w / h)
                    signals = {"eye_mediapipe": metrics.eye_closed, "eye_cnn": None,
                               "ear": metrics.ear, "ear3d": metrics.ear3d,
                               "mouth": metrics.mouth_open, "pitch_raw": metrics.pitch,
                               "yaw_raw": metrics.yaw, "roll_raw": metrics.roll,
                               "gaze_h": metrics.gaze_h, "distance_cm": metrics.distance_cm}
                    if cnn is not None:
                        p_closed = cnn.closed_probability(frame, face.landmarks)
                        if p_closed is not None:
                            signals["eye_cnn"] = p_closed
                            metrics.eye_closed = 0.5 * metrics.eye_closed + 0.5 * p_closed
                    signals["eye_fused"] = metrics.eye_closed
                    self.last_signals = signals
                else:
                    self.last_signals = {}

                snap = analyzer.update(now, metrics, len(faces))
                self.latest = snap
                self.analyzer = analyzer
                dt = now - last
                last = now
                if dt > 0:
                    self.fps = 0.9 * self.fps + 0.1 / dt if self.fps else 1.0 / dt

                if snap.alerts:
                    for a in snap.alerts:
                        self._alert_seq += 1
                        self._recent_alerts.append({
                            "id": self._alert_seq, "time": time.time(), "kind": a.kind,
                            "level": a.level, "message": a.message})
                    if recorder:
                        recorder.add_events(snap.alerts)
                    if self.sound_enabled:
                        worst = max(snap.alerts, key=lambda a: ("info", "warn", "alarm").index(a.level))
                        sound.play(worst.level)
                if self.on_update:
                    self.on_update(frame, snap, face)
        except Exception as exc:  # remonté à l'interface plutôt que de tuer le thread en silence
            if self.on_error:
                self.on_error(str(exc))
        finally:
            self.running = False
            if recorder and analyzer:
                recorder.close(analyzer, time.monotonic() - t_start)
            if cap is not None:
                cap.release()
            if detector is not None:
                detector.close()
