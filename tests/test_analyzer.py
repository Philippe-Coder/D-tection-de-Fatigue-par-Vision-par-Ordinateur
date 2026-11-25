"""Tests du cœur d'analyse : scénarios simulés, sans caméra."""
import numpy as np
import pytest

from fatigue_guard.analyzer import (ABSENT, ATTENTIF, CALIBRATION, DISTRAIT, SOMNOLENT,
                                    FatigueAnalyzer)
from fatigue_guard.config import PROFILES, get_profile
from fatigue_guard.features import (LEFT_EYE_EAR, FaceMetrics, euler_from_matrix,
                                    eye_aspect_ratio, metrics_from_face)

FPS = 30
DT = 1.0 / FPS


def face(eye=0.08, mouth=0.02, pitch=0.0, yaw=0.0, gaze=0.0, dist=60.0, roll=0.0) -> FaceMetrics:
    return FaceMetrics(eye_closed=eye, mouth_open=mouth, pitch=pitch, yaw=yaw, roll=roll,
                       gaze_h=gaze, distance_cm=dist)


class Sim:
    """Pilote un analyseur avec une horloge simulée."""

    def __init__(self, profile="bureau"):
        self.an = FatigueAnalyzer(get_profile(profile))
        self.t = 0.0
        self.alerts = []
        self.snap = None

    def run(self, seconds, metrics=None, n_faces=None, **kw):
        for _ in range(int(seconds * FPS)):
            m = metrics if metrics is not None or kw.get("absent") else face(**kw)
            if kw.get("absent"):
                m = None
            faces = n_faces if n_faces is not None else (0 if m is None else 1)
            self.snap = self.an.update(self.t, m, faces)
            self.alerts.extend(self.snap.alerts)
            self.t += DT
        return self.snap

    def kinds(self):
        return {a.kind for a in self.alerts}

    def calibrate(self):
        self.run(6)
        assert self.an.calibrated
        return self


def test_calibration_then_attentive():
    sim = Sim()
    assert sim.run(2).state == CALIBRATION
    snap = sim.run(6)
    assert snap.state == ATTENTIF
    assert snap.vigilance > 90
    assert not sim.alerts


def test_calibration_waits_for_a_stable_moment():
    """Cas mesuré : yeux fermés et tête en mouvement pendant les 2 premières secondes."""
    sim = Sim()
    for i in range(int(2.0 * FPS)):
        sim.snap = sim.an.update(sim.t, face(eye=0.6, pitch=14 - 6 * i / FPS, yaw=-5 + 4 * i / FPS), 1)
        sim.t += DT
    sim.run(3.5, pitch=-3, yaw=4)
    assert not sim.an.calibrated            # 5 s écoulées, mais le début n'était pas stable
    assert sim.snap.state == CALIBRATION
    sim.run(2.5, pitch=-3, yaw=4)
    assert sim.an.calibrated
    assert sim.an.base_pitch == pytest.approx(-3)
    assert sim.an.base_yaw == pytest.approx(4)
    assert sim.an.base_eye < 0.2


def test_calibration_gives_up_waiting_after_a_while():
    """Quelqu'un qui bouge sans arrêt finit quand même par être calibré."""
    sim = Sim()
    for i in range(int(22 * FPS)):
        sim.snap = sim.an.update(sim.t, face(yaw=20 * np.sin(i / FPS * 2.0)), 1)
        sim.t += DT
    assert sim.an.calibrated


def test_normal_blinks_are_counted_without_alarm():
    sim = Sim().calibrate()
    for _ in range(10):
        sim.run(3.0)
        sim.run(0.2, eye=0.85)  # clignement de 200 ms
    snap = sim.run(1)
    assert sim.an.total_blinks == 10
    assert 10 <= snap.blink_rate <= 25
    assert snap.state == ATTENTIF
    assert "microsleep" not in sim.kinds()


@pytest.mark.parametrize("profile", ["bureau", "conduite"])
def test_microsleep_triggers_alarm(profile):
    sim = Sim(profile).calibrate()
    limit = PROFILES[profile].microsleep_s
    sim.run(limit - 0.3, eye=0.9)
    assert "microsleep" not in sim.kinds()
    snap = sim.run(0.6, eye=0.9)
    assert "microsleep" in sim.kinds()
    assert snap.state == SOMNOLENT
    assert snap.alarm
    assert snap.vigilance <= 10


def test_driver_reacts_faster_than_desk():
    assert PROFILES["conduite"].microsleep_s < PROFILES["bureau"].microsleep_s
    assert PROFILES["conduite"].look_away_s < PROFILES["bureau"].look_away_s


def test_single_closure_early_in_session_is_not_fatigue():
    """Une fermeture isolée déclenche l'alarme, mais pas un état de fatigue durable."""
    sim = Sim().calibrate()
    sim.run(2.5, eye=0.9)
    assert "microsleep" in sim.kinds()
    snap = sim.run(4)
    assert snap.state == ATTENTIF
    assert snap.perclos < 0.15
    assert "fatigue" not in sim.kinds()


def test_repeated_long_closures_raise_fatigue():
    sim = Sim().calibrate()
    for _ in range(6):
        sim.run(2.5, eye=0.9)
        sim.run(3.0)
    assert sim.snap.fatigue > 35
    assert sim.snap.perclos > 0.2
    assert {"perclos", "microsleep"} <= sim.kinds()


def test_yawns_are_counted():
    sim = Sim().calibrate()
    for _ in range(3):
        sim.run(2.5, mouth=0.8)
        sim.run(4)
    assert sim.an.total_yawns == 3
    assert "yawn" in sim.kinds()


def test_talking_is_not_a_yawn():
    sim = Sim().calibrate()
    for _ in range(20):
        sim.run(0.3, mouth=0.7)
        sim.run(0.5, mouth=0.1)
    assert sim.an.total_yawns == 0


def test_looking_away_in_exam():
    sim = Sim("examen").calibrate()
    sim.run(2.0, yaw=40)
    assert "look_away" not in sim.kinds()
    snap = sim.run(2.0, yaw=40)
    assert "look_away" in sim.kinds()
    assert snap.state == DISTRAIT
    snap = sim.run(40)
    assert snap.state == ATTENTIF


def test_gaze_away_without_head_turn():
    sim = Sim("examen").calibrate()
    sim.run(4.0, gaze=0.8)
    assert "look_away" in sim.kinds()


def test_head_down_while_driving_is_alarm():
    sim = Sim("conduite").calibrate()
    sim.run(2.5, pitch=30)
    alarm = [a for a in sim.alerts if a.kind == "head_down"]
    assert alarm and alarm[0].level == "alarm"


# ------------------------------------------------------- postures de sommeil
DEEP_POSES = [({"pitch": -34}, "head_back"), ({"roll": 38}, "head_tilt"),
              ({"roll": -38}, "head_tilt")]
ALL_PROFILES = ["bureau", "ecran", "etude", "examen", "conduite"]


@pytest.mark.parametrize("pose, kind", DEEP_POSES)
@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_deep_posture_eyes_open_warns_fast_and_alarms_late(profile, pose, kind):
    """Tête franchement en arrière ou sur l'épaule, yeux nettement ouverts."""
    sim = Sim(profile).calibrate()
    limit = PROFILES[profile].posture_s
    sim.run(0.6 * limit - 0.4, **pose)
    assert kind not in sim.kinds()
    sim.run(0.8, **pose)
    assert kind in sim.kinds()
    sim.run(limit, **pose)
    assert "sleep_posture" not in sim.kinds()      # regarder le plafond un instant
    snap = sim.run(1.5 * limit + 1.0, **pose)
    assert "sleep_posture" in sim.kinds()          # …mais pas indéfiniment, immobile
    assert snap.state == SOMNOLENT and snap.alarm


@pytest.mark.parametrize("pose, kind", DEEP_POSES)
@pytest.mark.parametrize("profile", ALL_PROFILES)
def test_deep_posture_with_half_readable_closed_eyes_alarms_quickly(profile, pose, kind):
    """Yeux fermés mais mal lisibles (0,42 : sous le seuil normal de 0,45)."""
    sim = Sim(profile).calibrate()
    limit = PROFILES[profile].posture_s
    # ~1,3 s pour constater que la tête s'est immobilisée, puis un tiers du délai.
    snap = sim.run(limit / 3 + 2.0, eye=0.42, **pose)
    assert "microsleep" not in sim.kinds()         # les yeux seuls n'auraient pas suffi
    assert "sleep_posture" in sim.kinds()
    assert snap.state == SOMNOLENT


def test_head_back_like_real_recording():
    """Cas mesuré : angle bruité de ±10°, visage perdu 1 image sur 5, yeux flous."""
    rng = np.random.default_rng(7)
    sim = Sim("bureau").calibrate()
    for _ in range(int(4.0 * FPS)):
        if rng.random() < 0.2:
            m = None
        else:
            m = face(pitch=-40 + rng.uniform(-10, 10), eye=float(rng.uniform(0.33, 0.62)))
        sim.snap = sim.an.update(sim.t, m, 1 if m else 0)
        sim.alerts.extend(sim.snap.alerts)
        sim.t += DT
    assert {"sleep_posture", "microsleep", "sleep_lost"} & sim.kinds()
    assert sim.snap.state == SOMNOLENT


def test_brief_detection_gaps_do_not_reset_eye_closure():
    """Quelques images perdues ne doivent pas faire repartir le chrono à zéro."""
    sim = Sim("bureau").calibrate()
    for i in range(int(2.2 * FPS)):
        m = None if i % 6 == 0 else face(eye=0.9)
        sim.snap = sim.an.update(sim.t, m, 1 if m else 0)
        sim.alerts.extend(sim.snap.alerts)
        sim.t += DT
    assert "microsleep" in sim.kinds()


def test_real_yawn_level_is_counted():
    """Un vrai bâillement mesuré plafonne vers 0,45 d'ouverture."""
    sim = Sim().calibrate()
    sim.run(3.0, mouth=0.45)
    assert sim.an.total_yawns == 1


def test_head_back_with_eyes_closed_alarms_at_microsleep_delay():
    sim = Sim("bureau").calibrate()
    sim.run(2.0, pitch=-30, eye=0.9)
    assert "microsleep" in sim.kinds()
    assert sim.snap.state == SOMNOLENT


def test_slight_tilt_while_reading_never_alarms():
    """Tête appuyée sur la main, immobile : un avertissement au plus, pas d'alarme."""
    sim = Sim("bureau").calibrate()
    sim.run(60, roll=24)
    assert "head_tilt" in sim.kinds()
    assert "sleep_posture" not in sim.kinds()
    assert not any(a.level == "alarm" for a in sim.alerts)


def test_moving_head_in_deep_posture_is_not_sleep():
    """S'étirer ou balayer le plafond du regard n'est pas dormir."""
    sim = Sim("bureau").calibrate()
    for i in range(int(25 * FPS)):
        pitch = -44 + 14 * np.sin(i / FPS * 1.3)
        sim.snap = sim.an.update(sim.t, face(pitch=pitch), 1)
        sim.alerts.extend(sim.snap.alerts)
        sim.t += DT
    assert "sleep_posture" not in sim.kinds()


def test_slightly_reclined_never_alarms():
    """Adossé à sa chaise, tête un peu en arrière : jamais d'alarme."""
    sim = Sim("bureau").calibrate()
    sim.run(90, pitch=-22)
    assert "head_back" in sim.kinds()
    assert not any(a.level == "alarm" for a in sim.alerts)


def test_brief_head_tilt_is_ignored():
    sim = Sim("bureau").calibrate()
    for _ in range(5):
        sim.run(2.0, roll=38)
        sim.run(3.0)
    assert not sim.alerts


@pytest.mark.parametrize("before", [{"pitch": -30}, {"roll": 35}, {"eye": 0.9}, {"pitch": 28}])
def test_face_lost_after_sleep_sign_is_alarm_not_absence(before):
    """La tête bascule hors du champ : ce n'est pas un départ, c'est une chute."""
    sim = Sim("bureau").calibrate()
    sim.run(1.0, **before)
    snap = sim.run(3.5, absent=True)
    assert "sleep_lost" in sim.kinds()
    assert snap.state == SOMNOLENT
    snap = sim.run(20, absent=True)  # l'alarme ne sonne pas indéfiniment
    assert snap.state == ABSENT
    assert sim.run(1).state != SOMNOLENT


def test_leaving_normally_is_plain_absence():
    sim = Sim("bureau").calibrate()
    sim.run(3)
    snap = sim.run(12, absent=True)
    assert "sleep_lost" not in sim.kinds()
    assert snap.state == ABSENT


def test_absence():
    sim = Sim("examen").calibrate()
    snap = sim.run(5, absent=True)
    assert snap.state == ABSENT
    assert "absent" in sim.kinds()
    snap = sim.run(1)
    assert snap.state != ABSENT


def test_multiple_faces_only_in_exam():
    exam = Sim("examen").calibrate()
    exam.run(3, n_faces=2)
    assert "multi_face" in exam.kinds()
    desk = Sim("bureau").calibrate()
    desk.run(3, n_faces=2)
    assert "multi_face" not in desk.kinds()


def test_screen_profile_ignores_looking_away_but_warns_distance():
    sim = Sim("ecran").calibrate()
    sim.run(20, yaw=45, dist=24)
    assert "look_away" not in sim.kinds()
    assert "too_close" in sim.kinds()


def test_break_reminder():
    sim = Sim("ecran").calibrate()
    sim.run(20 * 60 + 5)
    assert "break" in sim.kinds()


def test_alert_cooldown_limits_repeats():
    sim = Sim("examen").calibrate()
    sim.run(30, yaw=40)
    assert 1 <= len([a for a in sim.alerts if a.kind == "look_away"]) <= 3


def test_calibration_adapts_to_narrow_eyes():
    """Une personne aux yeux naturellement peu ouverts ne doit pas alarmer."""
    sim = Sim()
    sim.run(6, eye=0.30)
    sim.run(30, eye=0.33)
    assert "microsleep" not in sim.kinds()
    assert sim.snap.perclos < 0.05


def test_snapshot_is_json_serialisable():
    import json

    sim = Sim().calibrate()
    snap = sim.run(3, eye=0.9)
    assert json.loads(json.dumps(snap.to_dict()))["state"] == SOMNOLENT


# ------------------------------------------------------------------ features
def test_ear_open_vs_closed():
    pts = np.zeros((478, 3))
    p1, p2, p3, p4, p5, p6 = LEFT_EYE_EAR
    pts[p1, :2], pts[p4, :2] = (0.0, 0.0), (1.0, 0.0)
    pts[p2, :2], pts[p6, :2] = (0.3, 0.15), (0.3, -0.15)
    pts[p3, :2], pts[p5, :2] = (0.7, 0.15), (0.7, -0.15)
    assert eye_aspect_ratio(pts, LEFT_EYE_EAR) == pytest.approx(0.3)
    pts[[p2, p3], 1] = 0.02
    pts[[p5, p6], 1] = -0.02
    assert eye_aspect_ratio(pts, LEFT_EYE_EAR) < 0.1


def test_eye_crop_is_straightened_when_head_is_tilted():
    """Tête penchée de 35° : le recadrage envoyé au CNN doit rester horizontal."""
    import cv2

    from fatigue_guard.eye_cnn import aligned_eye_crops
    from fatigue_guard.features import LEFT_EYE_LOOP, RIGHT_EYE_LOOP

    w, h = 640, 480
    lm = np.full((478, 3), 0.1)
    frame = np.zeros((h, w, 3), np.uint8)
    angle = np.radians(35)
    for (cx, cy), loop, (a, b) in (((220, 240), RIGHT_EYE_LOOP, (33, 133)),
                                   ((420, 240), LEFT_EYE_LOOP, (362, 263))):
        d = np.array([np.cos(angle), np.sin(angle)]) * 30
        pa, pb = np.array([cx, cy]) - d, np.array([cx, cy]) + d
        for i, k in enumerate(loop):  # points du contour répartis sur la ligne de l'œil
            lm[k, :2] = (pa + (pb - pa) * i / (len(loop) - 1)) / (w, h)
        lm[a, :2], lm[b, :2] = pa / (w, h), pb / (w, h)
        cv2.line(frame, tuple(pa.astype(int)), tuple(pb.astype(int)), (255, 255, 255), 3)

    crops = aligned_eye_crops(frame, lm)
    assert len(crops) == 2
    for crop in crops:
        ys, xs = np.nonzero(crop > 128)
        assert xs.max() - xs.min() > 30          # la ligne traverse le recadrage…
        assert ys.max() - ys.min() <= 6          # …et elle est horizontale
        assert abs(ys.mean() - crop.shape[0] / 2) < 3


def test_ear3d_resists_head_pitch():
    """L'EAR 3D reste stable quand la tête bascule, contrairement à l'EAR 2D."""
    pts = np.zeros((478, 3))
    p1, p2, p3, p4, p5, p6 = LEFT_EYE_EAR
    pts[p1, :2], pts[p4, :2] = (0.0, 0.0), (1.0, 0.0)
    pts[p2, :2], pts[p6, :2] = (0.3, 0.15), (0.3, -0.15)
    pts[p3, :2], pts[p5, :2] = (0.7, 0.15), (0.7, -0.15)
    a = np.radians(40)  # rotation autour de l'axe horizontal
    rot = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    tilted = pts @ rot.T
    assert eye_aspect_ratio(tilted, LEFT_EYE_EAR, 3) == pytest.approx(0.3)
    assert eye_aspect_ratio(tilted, LEFT_EYE_EAR, 2) < 0.25


def test_euler_angles():
    def rot_x(deg):
        a = np.radians(deg)
        return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])

    def rot_y(deg):
        a = np.radians(deg)
        return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])

    pitch, yaw, roll = euler_from_matrix(rot_x(25))
    assert (pitch, yaw, roll) == pytest.approx((25, 0, 0), abs=1e-6)
    pitch, yaw, roll = euler_from_matrix(rot_y(-30))
    assert (pitch, yaw, roll) == pytest.approx((0, -30, 0), abs=1e-6)


def test_metrics_use_blendshapes_and_distance():
    pts = np.random.default_rng(0).random((478, 3))
    matrix = np.eye(4)
    matrix[2, 3] = -55.0
    m = metrics_from_face(pts, {"eyeBlinkLeft": 0.8, "eyeBlinkRight": 0.6, "jawOpen": 0.4},
                          matrix)
    assert m.eye_closed == pytest.approx(0.7)
    assert m.mouth_open == pytest.approx(0.4)
    assert m.distance_cm == pytest.approx(55.0)
