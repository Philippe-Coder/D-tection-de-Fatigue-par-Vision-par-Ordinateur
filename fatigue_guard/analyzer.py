"""Analyse temporelle : transforme les mesures image par image en état de vigilance.

Ce module ne dépend ni de la caméra ni de l'interface : on lui fournit un
horodatage et des mesures, il renvoie un `Snapshot`. Il est donc testable
sans matériel.
"""
from __future__ import annotations

import statistics
from collections import deque
from dataclasses import asdict, dataclass, field

from .config import Profile
from .features import FaceMetrics

# États possibles
CALIBRATION = "CALIBRATION"
ATTENTIF = "ATTENTIF"
DISTRAIT = "DISTRAIT"
FATIGUE = "FATIGUE"
SOMNOLENT = "SOMNOLENT"
ABSENT = "ABSENT"

STATE_LABELS = {
    CALIBRATION: "Calibration…",
    ATTENTIF: "Attentif",
    DISTRAIT: "Distrait",
    FATIGUE: "Fatigue",
    SOMNOLENT: "Somnolence",
    ABSENT: "Absent",
}

CALIBRATION_S = 5.0
CALIB_MAX_S = 20.0       # au-delà, on calibre avec ce qu'on a
CALIB_SPREAD_DEG = 8.0   # dispersion tolérée de la tête pendant la calibration
WINDOW_S = 60.0          # fenêtre PERCLOS / clignements
PERCLOS_MIN_WINDOW_S = 20.0
ATTENTION_WINDOW_S = 30.0
EVENT_WINDOW_S = 300.0   # bâillements, hochements
BLINK_MIN_S = 0.04
BLINK_MAX_S = 0.6
YAWN_S = 1.5
# Mesuré sur un vrai bâillement : la bouche plafonne vers 0,30-0,55 ; la parole
# dépasse rarement 0,30 et jamais 1,5 s d'affilée.
YAWN_LEVEL = 0.32
BRIEF_LOSS_S = 0.6       # trou de détection toléré sans remettre les chronos à zéro
BREAK_RESET_S = 120.0    # absence qui compte comme une vraie pause
MAX_DT = 0.5
# La distance est une estimation (la focale de la webcam est supposée) : on
# garde un seuil prudent pour ne pas harceler l'utilisateur.
TOO_CLOSE_CM = 30.0

# Postures de sommeil (angles relatifs à la position calibrée).
HEAD_BACK_DEG = 18.0     # tête basculée en arrière
HEAD_TILT_DEG = 20.0     # tête penchée vers l'épaule
DEEP_BACK_DEG = 28.0     # angles francs : alarme si la posture est tenue, immobile
DEEP_TILT_DEG = 32.0
STILL_WINDOW_S = 2.5     # fenêtre d'observation du mouvement de la tête
# Tête en arrière, la mesure d'angle saute de ±10° d'une image à l'autre alors
# que la tête est immobile. On compare donc la médiane de la première moitié de
# la fenêtre à celle de la seconde : seul un vrai déplacement les sépare.
MOVING_SHIFT_DEG = 8.0
# Tête en arrière, la caméra voit les yeux par en dessous : fermés, ils ne
# donnent que ~0,45 au lieu de ~0,8. On abaisse donc le niveau « probablement
# fermés » de cette marge, uniquement pour pondérer la posture.
POSE_EYE_MARGIN = 0.08
EVIDENCE_CLOSED = 3.0    # vitesse du compteur de sommeil selon l'état des yeux
EVIDENCE_OPEN = 0.4
LOST_ALARM_MAX_S = 15.0  # durée max de l'alarme « visage perdu »
POSTURE_LABELS = {"back": "Tête en arrière", "tilt": "Tête penchée sur le côté",
                  "down": "Tête baissée"}

COOLDOWN = {"alarm": 2.5, "warn": 20.0, "info": 120.0}


@dataclass
class Alert:
    kind: str      # identifiant stable : microsleep, look_away, yawn…
    level: str     # info | warn | alarm
    message: str
    t: float = 0.0


@dataclass
class Snapshot:
    """État complet à un instant donné (sérialisable en JSON)."""

    t: float = 0.0
    profile: str = ""
    state: str = CALIBRATION
    state_label: str = STATE_LABELS[CALIBRATION]
    vigilance: float = 100.0
    fatigue: float = 0.0
    attention: float = 100.0
    perclos: float = 0.0
    blink_rate: float = 0.0
    blink_ms: float = 0.0
    yawns: int = 0
    eye_open: float = 1.0
    mouth_open: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    distance_cm: float = 0.0
    face_present: bool = False
    n_faces: int = 0
    eyes_closed_s: float = 0.0
    away_s: float = 0.0
    session_s: float = 0.0
    calibration: float = 0.0
    alarm: bool = False
    reason: str = ""
    alerts: list[Alert] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class _Sustained:
    """Durée pendant laquelle une condition reste vraie (tolère un bref trou)."""

    def __init__(self, grace: float = 0.25):
        self.grace = grace
        self.start: float | None = None
        self.last_true = 0.0

    def update(self, t: float, cond: bool) -> float:
        if cond:
            if self.start is None:
                self.start = t
            self.last_true = t
        elif self.start is not None and t - self.last_true > self.grace:
            self.start = None
        return 0.0 if self.start is None else t - self.start

    def reset(self) -> None:
        self.start = None


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


class FatigueAnalyzer:
    def __init__(self, profile: Profile):
        self.profile = profile
        self.t0: float | None = None
        self.last_t: float | None = None

        # Calibration
        self._calib: deque[tuple[float, FaceMetrics]] = deque()   # fenêtre glissante
        self._calib_time = 0.0
        self.calibrated = False
        self.base_eye = 0.1
        self.base_pitch = 0.0
        self.base_yaw = 0.0
        self.base_roll = 0.0

        # Fenêtres glissantes : (t, dt, yeux fermés, distrait, présent)
        self._frames: deque[tuple[float, float, bool, bool, bool]] = deque()
        self._blinks: deque[tuple[float, float]] = deque()      # (t, durée)
        self._long_closures: deque[float] = deque()
        self._yawns: deque[float] = deque()
        self._nods: deque[float] = deque()

        # Yeux
        self._eye_is_closed = False
        self._eye_hist: deque[float] = deque(maxlen=3)
        self._closure_start = 0.0
        self._closure_alarmed = False

        # Conditions maintenues
        self._yawn = _Sustained(0.3)
        self._yawn_counted = False
        self._head_down = _Sustained(0.4)
        self._nod_counted = False
        self._away = _Sustained(0.4)
        self._absent = _Sustained(0.0)
        self._multi = _Sustained(0.5)
        self._close = _Sustained(1.0)
        self._present_since: float | None = None

        # Postures de sommeil
        self._posture = _Sustained(0.5)
        self._posture_kind = ""
        self._sleep_counted = False
        self._sleep_evidence = 0.0    # secondes « équivalent sommeil » accumulées
        self._deep = False
        self._lost_in_posture = False
        self._pose_hist: deque[tuple[float, float, float, float]] = deque()
        self._last_risk_t = -1e9      # dernier instant « yeux fermés » ou posture anormale
        self._was_present = False
        self._lost_asleep = False     # visage perdu juste après un signe d'endormissement

        self._fatigue = 0.0
        self._last_alert: dict[str, float] = {}

        # Statistiques de session
        self.total_blinks = 0
        self.total_yawns = 0
        self.total_alarms = 0
        self.distracted_s = 0.0
        self.present_s = 0.0
        self._vig_sum = 0.0
        self._vig_time = 0.0
        self.min_vigilance = 100.0

    # ------------------------------------------------------------------ utils
    @property
    def mean_vigilance(self) -> float:
        return self._vig_sum / self._vig_time if self._vig_time > 0 else 100.0

    def recalibrate(self) -> None:
        self._calib.clear()
        self._calib_time = 0.0
        self.calibrated = False

    def _alert(self, out: list[Alert], t: float, kind: str, level: str, msg: str) -> None:
        last = self._last_alert.get(kind)
        if last is not None and t - last < COOLDOWN[level]:
            return
        self._last_alert[kind] = t
        if level == "alarm":
            self.total_alarms += 1
        out.append(Alert(kind, level, msg, t))

    @staticmethod
    def _calib_stable(frames: list[FaceMetrics]) -> bool:
        """Tête stable (dispersion 10e-90e centile) et yeux ouverts la plupart du temps."""
        def spread(values: list[float]) -> float:
            values = sorted(values)
            return values[int(0.9 * (len(values) - 1))] - values[int(0.1 * (len(values) - 1))]

        if any(spread([getattr(x, axis) for x in frames]) > CALIB_SPREAD_DEG
               for axis in ("pitch", "yaw", "roll")):
            return False
        return statistics.median(x.eye_closed for x in frames) < 0.4

    def _eye_thresholds(self) -> tuple[float, float]:
        on = min(0.70, max(0.45, self.base_eye + 0.35))
        return on, on - 0.15

    @staticmethod
    def _prune(dq: deque, t: float, window: float) -> None:
        while dq and t - (dq[0][0] if isinstance(dq[0], tuple) else dq[0]) > window:
            dq.popleft()

    # ----------------------------------------------------------------- update
    def update(self, t: float, m: FaceMetrics | None, n_faces: int = 0) -> Snapshot:
        if self.t0 is None:
            self.t0 = t
        dt = 0.0 if self.last_t is None else min(MAX_DT, max(0.0, t - self.last_t))
        self.last_t = t
        p = self.profile
        alerts: list[Alert] = []
        present = m is not None

        snap = Snapshot(t=t, profile=p.key, face_present=present, n_faces=n_faces,
                        session_s=t - self.t0)

        # ---------------------------------------------------------- absence
        absent_for = self._absent.update(t, not present)
        if not present:
            if self._was_present:
                # Le visage vient de disparaître. Si la tête basculait ou si les
                # yeux étaient fermés juste avant, ce n'est pas une absence :
                # la personne s'est probablement affaissée hors du champ.
                self._was_present = False
                self._lost_asleep = self.calibrated and t - self._last_risk_t <= 1.2
                self._lost_in_posture = self._lost_asleep and self._posture.start is not None
            if absent_for > BREAK_RESET_S:
                self._present_since = None
            if absent_for > BRIEF_LOSS_S:
                # Aux angles extrêmes le visage disparaît par à-coups : on ne
                # remet les chronos à zéro qu'après un vrai trou.
                self._eye_is_closed = False
                self._yawn.reset()
                self._head_down.reset()
                self._away.reset()
            if self._lost_in_posture and absent_for <= LOST_ALARM_MAX_S:
                # La posture est présumée tenue pendant que le visage est perdu.
                self._posture.update(t, True)
                if self._deep:
                    self._sleep_evidence += dt
            else:
                self._posture.reset()
                self._sleep_evidence = 0.0
            lost = (self._lost_asleep
                    and max(1.0, p.microsleep_s * 1.5) <= absent_for <= LOST_ALARM_MAX_S)
            asleep = self._lost_in_posture and self._sleep_evidence >= p.posture_s
            is_absent = p.absent_s > 0 and absent_for >= p.absent_s
            self._frames.append((t, dt, lost or asleep, is_absent or lost or asleep, False))
            self._prune(self._frames, t, WINDOW_S)
            if lost or asleep:
                self._alert(alerts, t, "sleep_lost", "alarm",
                            "Votre tête a basculé hors du champ — réveillez-vous !")
                return self._finish(snap, t, dt, alerts, distracted=True, absent=False,
                                    away_s=absent_for, reason="",
                                    critical="Tête basculée hors du champ")
            if is_absent:
                self._alert(alerts, t, "absent", p.absent_level,
                            "Visage non détecté — êtes-vous toujours là ?")
            return self._finish(snap, t, dt, alerts, distracted=is_absent,
                                absent=absent_for >= max(1.0, min(p.absent_s or 3.0, 3.0)),
                                away_s=absent_for, reason="Visage non détecté")

        self._was_present = True
        self._lost_asleep = False
        if self._present_since is None:
            self._present_since = t
        self.present_s += dt

        # ------------------------------------------------------ calibration
        if not self.calibrated:
            # Fenêtre glissante de 5 s : on ne calibre que sur un moment où la
            # tête est stable et les yeux ouverts. Sinon (personne qui s'installe,
            # yeux encore fermés) la référence serait faussée pour toute la session.
            self._calib.append((t, m))
            self._calib_time += dt
            while t - self._calib[0][0] > CALIBRATION_S:
                self._calib.popleft()
            frames = [x for _, x in self._calib]
            full = t - self._calib[0][0] >= CALIBRATION_S - 0.5 and len(frames) >= 15
            if full and (self._calib_stable(frames) or self._calib_time >= CALIB_MAX_S):
                eyes = sorted(x.eye_closed for x in frames)
                # Quartile bas : ignore les clignements pendant la calibration.
                self.base_eye = min(0.35, eyes[len(eyes) // 4])
                self.base_pitch = statistics.median(x.pitch for x in frames)
                self.base_yaw = statistics.median(x.yaw for x in frames)
                self.base_roll = statistics.median(x.roll for x in frames)
                self.calibrated = True
                self._calib.clear()
        if self.calibrated:
            snap.calibration = 1.0
        else:
            span = t - self._calib[0][0] if self._calib else 0.0
            snap.calibration = min(0.95, span / CALIBRATION_S)

        pitch = m.pitch - self.base_pitch
        yaw = m.yaw - self.base_yaw
        roll = m.roll - self.base_roll
        snap.pitch, snap.yaw, snap.roll = pitch, yaw, roll
        snap.distance_cm = m.distance_cm
        snap.mouth_open = m.mouth_open

        # ------------------------------------------------------------- yeux
        on, off = self._eye_thresholds()
        self._eye_hist.append(m.eye_closed)
        eye = statistics.median(self._eye_hist)
        snap.eye_open = _clip01(1.0 - (eye - self.base_eye) / max(0.2, on - self.base_eye))
        closed_now = eye > (off if self._eye_is_closed else on)
        if closed_now and not self._eye_is_closed:
            self._eye_is_closed = True
            self._closure_start = t
            self._closure_alarmed = False
        elif not closed_now and self._eye_is_closed:
            self._eye_is_closed = False
            dur = t - self._closure_start
            if BLINK_MIN_S <= dur <= BLINK_MAX_S:
                self._blinks.append((t, dur))
                self.total_blinks += 1
        closed_for = t - self._closure_start if self._eye_is_closed else 0.0
        snap.eyes_closed_s = closed_for

        if closed_for > 0.3:
            self._last_risk_t = t

        critical = ""
        microsleep = self.calibrated and closed_for >= p.microsleep_s
        if microsleep:
            critical = "Yeux fermés"
            if not self._closure_alarmed:
                self._closure_alarmed = True
                self._long_closures.append(t)
            self._alert(alerts, t, "microsleep", "alarm",
                        f"Yeux fermés depuis {closed_for:.1f} s — réveillez-vous !")

        # ------------------------------------------------------ bâillements
        yawn_for = self._yawn.update(t, m.mouth_open > YAWN_LEVEL)
        if yawn_for >= YAWN_S and not self._yawn_counted:
            self._yawn_counted = True
            self._yawns.append(t)
            self.total_yawns += 1
            if len(self._yawns) >= 3:
                self._alert(alerts, t, "yawn", "warn",
                            f"{len(self._yawns)} bâillements en 5 min — une pause s'impose.")
        elif yawn_for == 0.0:
            self._yawn_counted = False

        # ------------------------------------------------------- tête / regard
        head_down_for = away_for = 0.0
        distracted = False
        reason = ""
        if self.calibrated:
            if p.head_down_s > 0:
                head_down_for = self._head_down.update(t, pitch > p.pitch_deg)
                if head_down_for >= p.head_down_s:
                    distracted = True
                    reason = "Tête baissée"
                    if not self._nod_counted:
                        self._nod_counted = True
                        self._nods.append(t)
                    self._alert(alerts, t, "head_down", p.head_down_level,
                                "Tête baissée — redressez-vous et regardez devant vous.")
                elif head_down_for == 0.0:
                    self._nod_counted = False

            # --- Postures de sommeil : tête en arrière, sur l'épaule, baissée.
            # Actives dans tous les profils : ce sont des signes d'endormissement,
            # pas de simple distraction.
            self._pose_hist.append((t, pitch, yaw, roll))
            while self._pose_hist and t - self._pose_hist[0][0] > STILL_WINDOW_S:
                self._pose_hist.popleft()
            if pitch < -HEAD_BACK_DEG:
                kind = "back"
            elif abs(roll) > HEAD_TILT_DEG:
                kind = "tilt"
            elif pitch > p.pitch_deg:
                kind = "down"
            else:
                kind = ""
            if kind:
                self._posture_kind = kind
                self._last_risk_t = t
            # Angle franc : seule cette posture peut mener à l'alarme. Une tête
            # un peu penchée (appuyée sur la main) ne donne qu'un avertissement ;
            # la tête baissée garde sa règle par profil (livre posé sur la table).
            deep = pitch < -DEEP_BACK_DEG or abs(roll) > DEEP_TILT_DEG
            self._deep = deep
            posture_for = self._posture.update(t, bool(kind))
            if posture_for == 0.0:
                self._sleep_counted = False
                self._sleep_evidence = 0.0
            else:
                if deep and not self._is_moving():
                    # On accumule des « secondes de sommeil probable ». Les yeux
                    # sont mal lisibles sous ces angles : ils accélèrent ou
                    # ralentissent le compteur au lieu de décider seuls.
                    if eye > on - POSE_EYE_MARGIN:
                        rate = EVIDENCE_CLOSED      # yeux probablement fermés
                    elif eye < self.base_eye + 0.12:
                        rate = EVIDENCE_OPEN        # yeux nettement ouverts
                    else:
                        rate = 1.0                  # illisibles
                    self._sleep_evidence += dt * rate
                if posture_for >= p.posture_s * (0.6 if deep else 1.0):
                    label = POSTURE_LABELS[self._posture_kind]
                    distracted = True
                    reason = label
                    if self._posture_kind != "down" or p.head_down_s == 0:
                        self._alert(alerts, t, f"head_{self._posture_kind}", "warn",
                                    f"{label} — redressez-vous.")
                if self._sleep_evidence >= p.posture_s:
                    label = POSTURE_LABELS[self._posture_kind]
                    distracted = True
                    critical = critical or f"{label}, immobile"
                    if not self._sleep_counted:
                        self._sleep_counted = True
                        self._long_closures.append(t)
                    self._alert(alerts, t, "sleep_posture", "alarm",
                                f"{label} et immobile — vous vous endormez, réveillez-vous !")
            head_down_for = max(head_down_for, posture_for)

            if p.look_away_s > 0:
                head_away = abs(yaw) > p.yaw_deg
                gaze_away = (not self._eye_is_closed) and abs(m.gaze_h) > 0.6
                away_for = self._away.update(t, head_away or gaze_away)
                if away_for >= p.look_away_s:
                    distracted = True
                    reason = "Regard détourné"
                    self._alert(alerts, t, "look_away", p.look_away_level,
                                "Regard détourné — revenez à votre tâche.")

            if p.multi_face and self._multi.update(t, n_faces > 1) >= 2.0:
                distracted = True
                reason = "Plusieurs personnes"
                self._alert(alerts, t, "multi_face", "warn",
                            "Plusieurs visages détectés devant la caméra.")

            if p.eye_strain and m.distance_cm > 0:
                if self._close.update(t, m.distance_cm < TOO_CLOSE_CM) >= 15.0:
                    self._alert(alerts, t, "too_close", "info",
                                "Vous êtes très près de l'écran — reculez un peu.")

        snap.away_s = max(away_for, head_down_for)

        # -------------------------------------------------------- fenêtres
        self._frames.append((t, dt, self._eye_is_closed, distracted, True))
        for dq, w in ((self._frames, WINDOW_S), (self._blinks, WINDOW_S),
                      (self._long_closures, 120.0), (self._yawns, EVENT_WINDOW_S),
                      (self._nods, EVENT_WINDOW_S)):
            self._prune(dq, t, w)

        present_time = sum(f[1] for f in self._frames if f[4])
        closed_time = sum(f[1] for f in self._frames if f[4] and f[2])
        # En début de session la fenêtre est courte : on la borne à 20 s pour
        # qu'une seule fermeture d'yeux ne pèse pas comme une somnolence installée.
        perclos = closed_time / max(present_time, PERCLOS_MIN_WINDOW_S) if present_time > 2.0 else 0.0
        span = min(WINDOW_S, max(1.0, present_time))
        blink_rate = len(self._blinks) * 60.0 / span if present_time > 10.0 else 0.0
        blink_dur = statistics.fmean(b[1] for b in self._blinks) if self._blinks else 0.0
        snap.perclos = perclos
        snap.blink_rate = blink_rate
        snap.blink_ms = blink_dur * 1000.0
        snap.yawns = len(self._yawns)

        # ---------------------------------------------------- score fatigue
        if self.calibrated:
            raw = (
                0.40 * _clip01((perclos - 0.06) / 0.24)
                + 0.20 * _clip01(len(self._yawns) / 4.0)
                + 0.15 * _clip01((blink_dur - 0.18) / 0.25)
                + 0.10 * _clip01(len(self._nods) / 3.0)
                + 0.30 * _clip01(len(self._long_closures) / 2.0)
            )
            target = 100.0 * _clip01(raw)
            # Monte vite, redescend lentement.
            rate = 0.6 if target > self._fatigue else 0.08
            self._fatigue += (target - self._fatigue) * min(1.0, rate * dt)

            if perclos > 0.18 and present_time > 20.0:
                self._alert(alerts, t, "perclos", "warn",
                            f"Yeux fermés {perclos * 100:.0f} % du temps — signe net de fatigue.")
            if p.eye_strain and present_time > 50.0 and 0 < blink_rate < 7.0:
                self._alert(alerts, t, "low_blink", "info",
                            "Vous clignez peu des yeux — pensez à cligner et à regarder au loin.")
            if p.break_min > 0 and self._present_since is not None \
                    and t - self._present_since >= p.break_min * 60.0:
                self._present_since = t
                self._alert(alerts, t, "break", "info",
                            f"{p.break_min:.0f} min sans pause — levez-vous et regardez au loin 20 s.")

        return self._finish(snap, t, dt, alerts, distracted=distracted, absent=False,
                            away_s=snap.away_s, reason=reason, critical=critical)

    def _is_moving(self) -> bool:
        """Vrai si la tête s'est réellement déplacée pendant la fenêtre d'observation."""
        hist = self._pose_hist
        if len(hist) < 8:
            return False
        middle = (hist[0][0] + hist[-1][0]) / 2.0
        first = [h for h in hist if h[0] <= middle]
        second = [h for h in hist if h[0] > middle]
        if len(first) < 3 or len(second) < 3:
            return False
        for axis in (1, 2, 3):
            shift = statistics.median(h[axis] for h in first) - statistics.median(h[axis] for h in second)
            if abs(shift) > MOVING_SHIFT_DEG:
                return True
        return False

    # ----------------------------------------------------------------- finish
    def _finish(self, snap: Snapshot, t: float, dt: float, alerts: list[Alert], *,
                distracted: bool, absent: bool, away_s: float, reason: str,
                critical: str = "") -> Snapshot:
        # `critical` : motif d'un endormissement constaté (yeux fermés, posture
        # de sommeil, tête basculée hors champ) ; force l'état Somnolence.
        microsleep = bool(critical)
        if distracted:
            self.distracted_s += dt

        recent = [f for f in self._frames if t - f[0] <= ATTENTION_WINDOW_S]
        total = sum(f[1] for f in recent)
        bad = sum(f[1] for f in recent if f[3])
        attention = 100.0 * (1.0 - bad / total) if total > 1.0 else 100.0
        # Une distraction en cours pèse immédiatement.
        if distracted:
            attention = min(attention, 45.0)

        fatigue = self._fatigue
        vigilance = min(100.0 - fatigue, attention)
        if microsleep:
            vigilance = min(vigilance, 10.0)

        if absent:
            state = ABSENT
        elif not self.calibrated:
            state = CALIBRATION
        elif microsleep or fatigue >= 65.0:
            state = SOMNOLENT
        elif distracted:
            state = DISTRAIT
        elif fatigue >= 35.0:
            state = FATIGUE
        else:
            state = ATTENTIF

        if state == SOMNOLENT and not microsleep:
            self._alert(alerts, t, "drowsy", "alarm",
                        "Somnolence détectée — arrêtez-vous et faites une vraie pause.")
            reason = reason or "Somnolence"
        elif state == FATIGUE:
            self._alert(alerts, t, "fatigue", "warn",
                        "Signes de fatigue — prévoyez une pause prochainement.")
            reason = reason or "Signes de fatigue"
        if critical:
            reason = critical

        snap.state = state
        snap.state_label = STATE_LABELS[state]
        snap.fatigue = fatigue
        snap.attention = attention
        snap.vigilance = max(0.0, vigilance)
        snap.away_s = away_s
        snap.reason = reason
        snap.alerts = alerts
        snap.alarm = any(a.level == "alarm" for a in alerts) or microsleep

        if self.calibrated and state != ABSENT:
            self._vig_sum += snap.vigilance * dt
            self._vig_time += dt
            self.min_vigilance = min(self.min_vigilance, snap.vigilance)
        return snap
