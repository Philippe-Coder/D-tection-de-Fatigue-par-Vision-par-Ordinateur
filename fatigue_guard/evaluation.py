"""Test guidé : mesure la détection posture par posture sur votre visage.

`fatigue-guard evaluate` vous demande de prendre une série de poses (yeux
fermés, tête en arrière, tête sur l'épaule…) et enregistre les MESURES
(angles, ouverture des yeux, alertes) dans un fichier CSV. Aucune image
n'est enregistrée. Le tableau final montre ce qui est détecté et ce qui est
manqué, et le fichier sert à régler les seuils sur des données réelles.
"""
from __future__ import annotations

import csv
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import alerts as sound
from .config import DATA_DIR

PREPARE_S = 5.0
HOLD_S = 8.0
EARLY_S = 3.0
SLEEP_ALERTS = ("microsleep", "sleep_posture", "sleep_lost")


@dataclass(frozen=True)
class Step:
    key: str
    label: str
    instruction: str
    expected: tuple[str, ...]   # une de ces alertes doit apparaître ; () = aucune alarme
    eyes_closed: bool = False


STEPS = [
    Step("reference", "Référence", "Regardez l'écran normalement, yeux ouverts", ()),
    Step("yeux_fermes", "Yeux fermés", "Tête droite, FERMEZ les yeux", ("microsleep",), True),
    Step("arriere_ouverts", "Arrière, yeux ouverts", "Tête EN ARRIÈRE (regard vers le plafond), yeux ouverts",
         ("head_back",)),
    Step("arriere_fermes", "Arrière, yeux fermés", "Tête EN ARRIÈRE, yeux FERMÉS (comme endormi sur une chaise)",
         SLEEP_ALERTS, True),
    Step("epaule_gauche_ouverts", "Épaule G, yeux ouverts", "Tête penchée sur l'épaule GAUCHE, yeux ouverts",
         ("head_tilt",)),
    Step("epaule_gauche_fermes", "Épaule G, yeux fermés", "Tête penchée sur l'épaule GAUCHE, yeux FERMÉS",
         SLEEP_ALERTS, True),
    Step("epaule_droite_fermes", "Épaule D, yeux fermés", "Tête penchée sur l'épaule DROITE, yeux FERMÉS",
         SLEEP_ALERTS, True),
    Step("baissee_fermes", "Baissée, yeux fermés", "Menton vers la poitrine (tête baissée), yeux FERMÉS",
         SLEEP_ALERTS + ("head_down",), True),
    Step("baillement", "Bâillement", "BÂILLEZ longuement, bouche grande ouverte", ("yawn_counted",)),
    Step("tete_tournee", "Tête tournée", "Tournez la tête vers la GAUCHE (regard hors écran)", ("look_away",)),
]

FIELDS = ["t", "step", "phase", "face", "eye_mediapipe", "eye_cnn", "eye_fused", "ear", "ear3d",
          "mouth", "pitch_raw", "yaw_raw", "roll_raw", "pitch", "yaw", "roll", "gaze_h",
          "distance_cm", "eyes_closed_s", "perclos", "state", "vigilance", "alerts"]


def _median(rows: list[dict], key: str):
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return statistics.median(values) if values else None


def _fmt(value, pattern: str = "{:.2f}") -> str:
    return "—" if value is None else pattern.format(value)


def summarize(rows: list[dict]) -> list[dict]:
    """Résumé par étape, à partir des lignes de la phase de maintien."""
    summary = []
    for step in STEPS:
        hold = [r for r in rows if r["step"] == step.key and r["phase"] == "hold"]
        if not hold:
            continue
        seen = [r for r in hold if r["face"]]
        # On prend souvent la pose avant le bip : les alertes des dernières
        # secondes de préparation comptent aussi.
        start = float(hold[0]["t"])
        early = [r for r in rows if r["step"] == step.key and r["phase"] == "prepare"
                 and start - float(r["t"]) <= EARLY_S]
        window = early + hold if step.expected else hold
        kinds = sorted({k for r in window for k in r["alerts"].split("|") if k})
        delay = None
        if step.expected:
            detected = any(k in kinds for k in step.expected)
            hits = [r["t"] for r in window if any(k in r["alerts"].split("|") for k in step.expected)]
            if hits:
                delay = max(0.0, float(hits[0]) - start)
        else:
            detected = not any(k in kinds for k in SLEEP_ALERTS)
        summary.append({
            "step": step.key, "label": step.label,
            "face_pct": 100.0 * len(seen) / len(hold),
            "eye_mediapipe": _median(seen, "eye_mediapipe"), "eye_cnn": _median(seen, "eye_cnn"),
            "ear3d": _median(seen, "ear3d"), "pitch": _median(seen, "pitch"),
            "yaw": _median(seen, "yaw"), "roll": _median(seen, "roll"),
            "alerts": kinds, "detected": detected, "delay": delay,
        })
    return summary


def run(profile: str = "etude", camera: int = 0, use_cnn: bool = True) -> int:
    from rich.console import Console

    from .engine import Engine

    console = Console()
    rows: list[dict] = []
    errors: list[str] = []
    current = {"step": "attente", "phase": "idle"}
    t0 = time.monotonic()

    def on_update(_frame, snap, face):
        s = engine.last_signals if face is not None else {}
        row = {"t": round(time.monotonic() - t0, 3), "step": current["step"],
               "phase": current["phase"], "face": int(face is not None),
               "pitch": snap.pitch, "yaw": snap.yaw, "roll": snap.roll,
               "eyes_closed_s": snap.eyes_closed_s, "perclos": snap.perclos,
               "state": snap.state, "vigilance": snap.vigilance,
               "alerts": "|".join(a.kind for a in snap.alerts)}
        for key in ("eye_mediapipe", "eye_cnn", "eye_fused", "ear", "ear3d", "mouth",
                    "pitch_raw", "yaw_raw", "roll_raw", "gaze_h", "distance_cm"):
            row[key] = s.get(key)
        rows.append(row)

    engine = Engine(profile, camera, sound_enabled=False, record=False, use_cnn=use_cnn,
                    on_update=on_update, on_error=errors.append)

    console.print("\n[bold]Test guidé de Fatigue Guard[/] — environ 2 minutes.\n"
                  "Pour chaque pose : lisez la consigne, prenez la pose au [bold]bip[/], "
                  "et gardez-la jusqu'au [bold]double bip[/] (utile quand vos yeux sont fermés).\n"
                  "Seules des mesures chiffrées sont enregistrées, aucune image.\n")
    engine.start()
    try:
        with console.status("Démarrage de la caméra…"):
            while not engine.running and not errors:
                time.sleep(0.1)
        if errors:
            console.print(f"[bold red]Erreur :[/] {errors[0]}")
            return 1

        current.update(step="calibration", phase="hold")
        with console.status("Calibration : regardez l'écran normalement…"):
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and not errors:
                if engine.analyzer is not None and engine.analyzer.calibrated:
                    break
                time.sleep(0.1)
        if engine.analyzer is None or not engine.analyzer.calibrated:
            console.print("[bold red]Calibration impossible :[/] visage non détecté.")
            return 1

        for number, step in enumerate(STEPS, 1):
            if errors:
                break
            console.print(f"[bold cyan]{number}/{len(STEPS)}[/]  [bold]{step.instruction}[/]")
            current.update(step=step.key, phase="prepare")
            yawns_before = engine.analyzer.total_yawns
            end = time.monotonic() + PREPARE_S
            with console.status("") as status:
                while time.monotonic() < end:
                    status.update(f"Préparez-vous… {end - time.monotonic():.0f} s")
                    time.sleep(0.1)
            sound.play("info")
            current["phase"] = "hold"
            end = time.monotonic() + HOLD_S
            with console.status("") as status:
                while time.monotonic() < end:
                    status.update(f"[bold]Gardez la pose[/]… {end - time.monotonic():.0f} s")
                    time.sleep(0.1)
            if engine.analyzer.total_yawns > yawns_before and rows:
                rows[-1]["alerts"] = "|".join(filter(None, [rows[-1]["alerts"], "yawn_counted"]))
            current["phase"] = "rest"
            sound.play("warn")
            time.sleep(1.0)
    except KeyboardInterrupt:
        console.print("Test interrompu.")
    finally:
        engine.stop()

    if errors:
        console.print(f"[bold red]Erreur :[/] {errors[0]}")
        return 1

    folder = DATA_DIR / "evaluations"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"evaluation_{datetime.now():%Y%m%d_%H%M%S}.csv"
    write_csv(path, rows)

    print_summary(console, summarize(rows))
    console.print(f"Mesures enregistrées dans : [bold]{path}[/]")
    return 0


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in row.items()})


def print_summary(console, summary: list[dict], title: str = "Résultat par posture") -> None:
    from rich.table import Table

    table = Table(title=title)
    table.add_column("Posture", no_wrap=True)
    for col in ("Visage", "Yeux MP", "Yeux CNN", "Haut-bas", "Épaule", "Délai"):
        table.add_column(col, justify="right", no_wrap=True)
    table.add_column("Alertes")
    table.add_column("Verdict", no_wrap=True)
    for s in summary:
        table.add_row(
            s["label"], f"{s['face_pct']:.0f} %", _fmt(s["eye_mediapipe"]),
            _fmt(s["eye_cnn"]), _fmt(s["pitch"], "{:+.0f}°"), _fmt(s["roll"], "{:+.0f}°"),
            _fmt(s["delay"], "{:.1f} s"), ", ".join(s["alerts"]) or "—",
            "[green]détecté[/]" if s["detected"] else "[bold red]MANQUÉ[/]")
    console.print(table)
    missed = sum(1 for s in summary if not s["detected"])
    console.print(f"{len(summary) - missed}/{len(summary)} postures correctement traitées.")


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def replay_rows(path: Path, profile: str = "etude", fusion=None) -> list[dict]:
    """Rejoue un enregistrement dans l'analyseur actuel.

    Les signaux bruts du fichier (yeux, bouche, angles) sont réinjectés image
    par image : on mesure ainsi l'effet d'un changement de seuil sur des
    données réelles, sans refaire le test devant la caméra. `fusion(mp, cnn)`
    permet d'essayer une autre combinaison des deux lecteurs d'yeux.
    """
    from .analyzer import FatigueAnalyzer
    from .config import get_profile
    from .features import FaceMetrics

    analyzer = FatigueAnalyzer(get_profile(profile))
    out = []
    step, yawns = None, 0
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["step"] != step:
                step, yawns = row["step"], analyzer.total_yawns
            metrics = None
            if row["face"] == "1":
                mp, cnn = _num(row["eye_mediapipe"]), _num(row["eye_cnn"])
                if fusion is not None and mp is not None and cnn is not None:
                    eye = fusion(mp, cnn)
                else:
                    eye = _num(row["eye_fused"]) or 0.0
                metrics = FaceMetrics(
                    eye_closed=eye, mouth_open=_num(row["mouth"]) or 0.0,
                    pitch=_num(row["pitch_raw"]) or 0.0, yaw=_num(row["yaw_raw"]) or 0.0,
                    roll=_num(row["roll_raw"]) or 0.0, gaze_h=_num(row["gaze_h"]) or 0.0,
                    distance_cm=_num(row["distance_cm"]) or 0.0)
            snap = analyzer.update(float(row["t"]), metrics, 1 if metrics else 0)
            kinds = [a.kind for a in snap.alerts]
            if row["phase"] == "hold" and analyzer.total_yawns > yawns:
                yawns = analyzer.total_yawns
                kinds.append("yawn_counted")
            new = dict(row)
            new.update(face=int(row["face"]), alerts="|".join(kinds), state=snap.state,
                       pitch=snap.pitch, yaw=snap.yaw, roll=snap.roll,
                       eye_mediapipe=_num(row["eye_mediapipe"]), eye_cnn=_num(row["eye_cnn"]),
                       ear3d=_num(row["ear3d"]), t=float(row["t"]))
            out.append(new)
    return out


def replay(path: str, profile: str = "etude") -> int:
    from rich.console import Console

    file = Path(path)
    if not file.exists():
        print(f"Fichier introuvable : {file}")
        return 1
    console = Console()
    print_summary(console, summarize(replay_rows(file, profile)),
                  title=f"Rejeu de {file.name} (profil {profile})")
    return 0
