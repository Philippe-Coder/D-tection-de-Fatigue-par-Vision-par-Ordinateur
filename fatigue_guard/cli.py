"""Ligne de commande : `fatigue-guard [gui|run|stats|train|profiles|doctor]`."""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

from . import __version__
from .config import DEFAULT_PORT, DEFAULT_PROFILE, PROFILES


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-p", "--profile", choices=list(PROFILES), default=DEFAULT_PROFILE,
                        help="profil d'utilisation (défaut : %(default)s)")
    parser.add_argument("-c", "--camera", type=int, default=0, help="numéro de la caméra")
    parser.add_argument("--no-sound", action="store_true", help="désactive les alertes sonores")
    parser.add_argument("--no-cnn", action="store_true",
                        help="n'utilise pas le CNN des yeux, même s'il est entraîné")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help="port du serveur d'état local pour les extensions")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fatigue-guard",
        description="Détection de fatigue et de distraction par webcam, 100 %% locale.")
    parser.add_argument("--version", action="version", version=f"fatigue-guard {__version__}")
    sub = parser.add_subparsers(dest="command")

    gui = sub.add_parser("gui", help="tableau de bord graphique (par défaut)")
    _common(gui)
    gui.add_argument("--start", action="store_true", help="démarre la surveillance tout de suite")
    gui.add_argument("--minimized", action="store_true",
                     help="démarre réduit dans la zone de notification")

    run = sub.add_parser("run", help="surveillance dans le terminal, sans fenêtre")
    _common(run)

    stats = sub.add_parser("stats", help="historique des dernières sessions")
    stats.add_argument("-n", type=int, default=15, help="nombre de sessions")

    train = sub.add_parser("train", help="entraîne le CNN d'état des yeux")
    train.add_argument("--data", default="data", help="dossier des données")
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--limit", type=int, default=0,
                       help="limite le nombre d'images (essai rapide)")

    evaluate = sub.add_parser(
        "evaluate", help="test guidé : mesure la détection posture par posture (2 min)")
    evaluate.add_argument("-p", "--profile", choices=list(PROFILES), default="etude")
    evaluate.add_argument("-c", "--camera", type=int, default=0)
    evaluate.add_argument("--no-cnn", action="store_true")
    evaluate.add_argument("--replay", metavar="CSV",
                          help="rejoue un enregistrement dans le code actuel, sans caméra")

    sub.add_parser("profiles", help="liste les profils disponibles")
    doctor = sub.add_parser("doctor", help="vérifie la caméra et les modèles")
    doctor.add_argument("-c", "--camera", type=int, default=0)
    return parser


# ------------------------------------------------------------------ commandes
def cmd_run(args) -> int:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress_bar import ProgressBar
    from rich.table import Table
    from rich.text import Text

    from .engine import Engine
    from .server import StatusServer

    console = Console()
    styles = {"ATTENTIF": "green", "DISTRAIT": "yellow", "FATIGUE": "yellow",
              "SOMNOLENT": "bold red", "ABSENT": "dim", "CALIBRATION": "cyan"}
    level_styles = {"info": "cyan", "warn": "yellow", "alarm": "bold red"}
    log: list[Text] = []
    errors: list[str] = []
    status = {"msg": "Démarrage…"}

    def on_update(_frame, snap, _face):
        for a in snap.alerts:
            line = Text(datetime.now().strftime("%H:%M:%S  "), style="dim")
            line.append(a.message, style=level_styles.get(a.level, ""))
            log.insert(0, line)
            del log[8:]

    engine = Engine(args.profile, args.camera, sound_enabled=not args.no_sound,
                    use_cnn=not args.no_cnn, on_update=on_update, on_error=errors.append,
                    on_status=lambda m: status.update(msg=m))
    server = StatusServer(engine.status, args.port)
    server.start()

    def render():
        s = engine.latest
        if not engine.running:
            return Panel(Text(status["msg"] or "Démarrage…", style="cyan"), title="Fatigue Guard")
        style = styles.get(s.state, "")
        color = "green" if s.vigilance >= 70 else "yellow" if s.vigilance >= 40 else "red"
        head = Table.grid(padding=(0, 2))
        head.add_row(Text(s.state_label.upper(), style=style),
                     ProgressBar(total=100, completed=s.vigilance, width=30,
                                 complete_style=color, finished_style=color),
                     Text(f"{s.vigilance:3.0f} %", style=f"bold {color}"))
        if s.state == "CALIBRATION":
            detail = f"Regardez l'écran normalement… {s.calibration * 100:.0f} %"
        else:
            detail = s.reason or "Tout va bien"
        grid = Table.grid(padding=(0, 3))
        grid.add_row("Yeux fermés", f"{s.perclos * 100:.0f} %", "Clignements",
                     f"{s.blink_rate:.0f}/min", "Bâillements", str(s.yawns))
        grid.add_row("Tête", f"{s.yaw:+.0f}° / {s.pitch:+.0f}°", "Distance",
                     f"{s.distance_cm:.0f} cm" if s.distance_cm else "—",
                     "Session", f"{int(s.session_s) // 60} min {int(s.session_s) % 60:02d}")
        alerts = Group(*log) if log else Text("Aucune alerte", style="dim")
        body = Group(head, Text(detail, style="dim"), Text(""), grid, Text(""),
                     Text("Alertes", style="bold"), alerts)
        return Panel(body, title=f"Fatigue Guard · {engine.profile.label}",
                     subtitle=f"{engine.fps:.0f} images/s · Ctrl+C pour arrêter", border_style=style)

    engine.start()
    try:
        with Live(render(), console=console, refresh_per_second=8, screen=False) as live:
            while not errors:
                time.sleep(0.12)
                live.update(render())
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        server.stop()
    if errors:
        console.print(f"[bold red]Erreur :[/] {errors[0]}")
        return 1
    console.print("Session terminée. Historique : [bold]fatigue-guard stats[/]")
    return 0


def cmd_stats(args) -> int:
    from rich.console import Console
    from rich.table import Table

    from .storage import recent_sessions

    sessions = recent_sessions(args.n)
    console = Console()
    if not sessions:
        console.print("Aucune session enregistrée pour l'instant.")
        return 0
    table = Table(title="Dernières sessions")
    for col in ("Date", "Profil", "Durée", "Vigilance moy.", "Min.", "Distrait",
                "Bâillements", "Alarmes"):
        table.add_column(col, justify="right" if col not in ("Date", "Profil") else "left")
    for s in sessions:
        profile = PROFILES.get(s["profile"])
        table.add_row(
            datetime.fromtimestamp(s["started"]).strftime("%d/%m/%Y %H:%M"),
            profile.label if profile else s["profile"],
            f"{int(s['duration_s']) // 60} min",
            f"{s['avg_vigilance']:.0f} %", f"{s['min_vigilance']:.0f} %",
            f"{int(s['distracted_s'])} s", str(s["yawns"]), str(s["alarms"]))
    console.print(table)
    return 0


def cmd_profiles(_args) -> int:
    for p in PROFILES.values():
        print(f"{p.key:<10} {p.label:<24} {p.description}")
    return 0


def cmd_doctor(args) -> int:
    import cv2

    from .detector import FaceDetector
    from .engine import open_camera
    from .eye_cnn import find_model

    ok = True
    print(f"fatigue-guard {__version__} · Python {sys.version.split()[0]} · OpenCV {cv2.__version__}")
    try:
        detector = FaceDetector(progress=print)
        print("[OK] Modèle de visage chargé")
    except Exception as exc:
        print(f"[ERREUR] Modèle de visage : {exc}")
        return 1
    cnn = find_model()
    print(f"[OK] CNN des yeux : {cnn}" if cnn else
          "[--] CNN des yeux non entraîné (facultatif : fatigue-guard train)")
    cap = open_camera(args.camera)
    if cap is None:
        print(f"[ERREUR] Caméra {args.camera} inaccessible")
        ok = False
    else:
        found = 0
        t0 = time.monotonic()
        frames = 0
        while time.monotonic() - t0 < 3.0:
            good, frame = cap.read()
            if not good:
                continue
            frames += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if detector.detect(rgb, int((time.monotonic() - t0) * 1000)):
                found += 1
        cap.release()
        h, w = frame.shape[:2] if frames else (0, 0)
        print(f"[OK] Caméra {args.camera} : {w}x{h}, {frames / 3.0:.0f} images/s analysées")
        print(f"[{'OK' if found else '--'}] Visage détecté sur {found}/{frames} images")
    detector.close()
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = args.command or "gui"
    if command == "gui":
        from .gui import run_gui

        return run_gui(
            profile=getattr(args, "profile", DEFAULT_PROFILE),
            camera=getattr(args, "camera", 0),
            sound=not getattr(args, "no_sound", False),
            port=getattr(args, "port", DEFAULT_PORT),
            autostart=getattr(args, "start", False),
            minimized=getattr(args, "minimized", False),
            use_cnn=not getattr(args, "no_cnn", False),
        )
    if command == "run":
        return cmd_run(args)
    if command == "stats":
        return cmd_stats(args)
    if command == "profiles":
        return cmd_profiles(args)
    if command == "doctor":
        return cmd_doctor(args)
    if command == "evaluate":
        from .evaluation import replay, run

        if args.replay:
            return replay(args.replay, args.profile)
        return run(args.profile, args.camera, use_cnn=not args.no_cnn)
    if command == "train":
        from .training import train

        return train(args.data, epochs=args.epochs, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
