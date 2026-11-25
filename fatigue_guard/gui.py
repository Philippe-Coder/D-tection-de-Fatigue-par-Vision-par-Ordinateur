"""Interface graphique (PySide6) : tableau de bord temps réel."""
from __future__ import annotations

import math
import sys
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from . import __version__
from .analyzer import (ABSENT, ATTENTIF, CALIBRATION, DISTRAIT, FATIGUE, HEAD_BACK_DEG,
                       HEAD_TILT_DEG, SOMNOLENT, TOO_CLOSE_CM)
from .config import APP_NAME, DEFAULT_PORT, PROFILES
from .engine import Engine
from .features import FACE_OVAL, LEFT_EYE_LOOP, LIPS_LOOP, RIGHT_EYE_LOOP
from .server import StatusServer
from .storage import recent_sessions

# ------------------------------------------------------------------ palette
BG = "#0B0F17"
CARD = "#121926"
CARD_2 = "#182131"
BORDER = "#222D40"
TEXT = "#E8EEF8"
MUTED = "#8B98AD"
ACCENT = "#5B8DEF"
OK = "#34D399"
WARN = "#FBBF24"
DANGER = "#FB5C74"
VIOLET = "#A78BFA"

STATE_COLORS = {
    CALIBRATION: ACCENT,
    ATTENTIF: OK,
    DISTRAIT: WARN,
    FATIGUE: WARN,
    SOMNOLENT: DANGER,
    ABSENT: MUTED,
}
LEVEL_COLORS = {"info": ACCENT, "warn": WARN, "alarm": DANGER}

QSS = f"""
* {{ font-family: "Segoe UI Variable Text", "Segoe UI", "Inter", sans-serif; color: {TEXT}; }}
QMainWindow, QDialog {{ background: {BG}; }}
QFrame#card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px; }}
QLabel#title {{ font-size: 19px; font-weight: 700; }}
QLabel#subtitle, QLabel#muted {{ color: {MUTED}; font-size: 12px; }}
QLabel#cardTitle {{ color: {MUTED}; font-size: 11px; font-weight: 600; letter-spacing: 1px; }}
QLabel#tileValue {{ font-size: 22px; font-weight: 700; }}
QLabel#tileHint {{ color: {MUTED}; font-size: 11px; }}
QLabel#state {{ font-size: 24px; font-weight: 700; }}
QPushButton {{
    background: {CARD_2}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 8px 16px; font-size: 13px; font-weight: 600;
}}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton#primary {{ background: {ACCENT}; border: none; color: #08101F; padding: 9px 22px; }}
QPushButton#primary:hover {{ background: #7BA5F5; }}
QPushButton#primary[running="true"] {{ background: {CARD_2}; color: {DANGER}; border: 1px solid {DANGER}; }}
QPushButton:checked {{ border-color: {ACCENT}; color: {ACCENT}; }}
QComboBox {{
    background: {CARD_2}; border: 1px solid {BORDER}; border-radius: 10px;
    padding: 7px 12px; font-size: 13px; min-height: 20px; min-width: 96px;
}}
QComboBox:hover {{ border-color: {ACCENT}; }}
QComboBox:disabled {{ color: {MUTED}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox QAbstractItemView {{
    background: {CARD_2}; border: 1px solid {BORDER}; selection-background-color: {BORDER};
    outline: none; padding: 4px;
}}
QListWidget {{ background: transparent; border: none; font-size: 12px; outline: none; }}
QListWidget::item {{ padding: 7px 4px; border-bottom: 1px solid {BORDER}; }}
QTableWidget {{
    background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px;
    gridline-color: {BORDER}; font-size: 12px;
}}
QHeaderView::section {{
    background: {CARD_2}; color: {MUTED}; border: none; padding: 8px; font-weight: 600;
}}
QScrollBar:vertical {{ background: transparent; width: 8px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QToolTip {{ background: {CARD_2}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px; }}
"""


def _bgr(hex_color: str) -> tuple[int, int, int]:
    c = QtGui.QColor(hex_color)
    return c.blue(), c.green(), c.red()


def make_icon(color: str = ACCENT, size: int = 64) -> QtGui.QIcon:
    """Icône (œil stylisé) dessinée à la volée, teintée selon l'état."""
    pix = QtGui.QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QtGui.QPainter(pix)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.scale(size / 64.0, size / 64.0)
    p.setBrush(QtGui.QColor(BG))
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(2, 2, 60, 60, 16, 16)
    path = QtGui.QPainterPath()
    path.moveTo(10, 32)
    path.quadTo(32, 8, 54, 32)
    path.quadTo(32, 56, 10, 32)
    p.setPen(QtGui.QPen(QtGui.QColor(color), 4))
    p.setBrush(Qt.NoBrush)
    p.drawPath(path)
    p.setBrush(QtGui.QColor(color))
    p.setPen(Qt.NoPen)
    p.drawEllipse(QtCore.QPointF(32, 32), 8, 8)
    p.end()
    return QtGui.QIcon(pix)


# ------------------------------------------------------------------ widgets
class Card(QtWidgets.QFrame):
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.body = QtWidgets.QVBoxLayout(self)
        self.body.setContentsMargins(18, 16, 18, 16)
        self.body.setSpacing(10)
        if title:
            label = QtWidgets.QLabel(title.upper())
            label.setObjectName("cardTitle")
            self.body.addWidget(label)


class VideoView(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self._image: QtGui.QImage | None = None
        self._message = "Cliquez sur « Démarrer » pour lancer la surveillance"
        self._alarm_until = 0.0
        self._alarm_text = ""
        self._border = BORDER

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        self._image = QtGui.QImage(frame_bgr.data, w, h, 3 * w, QtGui.QImage.Format_BGR888).copy()
        self.update()

    def set_message(self, text: str) -> None:
        self._image = None
        self._message = text
        self.update()

    def set_border(self, color: str) -> None:
        self._border = color

    def flash_alarm(self, text: str) -> None:
        self._alarm_text = text
        self._alarm_until = time.monotonic() + 2.5

    def paintEvent(self, _event) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        rect = QtCore.QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        clip = QtGui.QPainterPath()
        clip.addRoundedRect(rect, 16, 16)
        p.fillPath(clip, QtGui.QColor("#070A10"))

        if self._image is not None:
            p.save()
            p.setClipPath(clip)
            # Remplit la carte en conservant les proportions (recadrage centré).
            iw, ih = self._image.width(), self._image.height()
            scale = max(rect.width() / iw, rect.height() / ih)
            w, h = iw * scale, ih * scale
            target = QtCore.QRectF(rect.center().x() - w / 2, rect.center().y() - h / 2, w, h)
            p.drawImage(target, self._image)
            p.restore()
        else:
            p.setPen(QtGui.QColor(MUTED))
            font = p.font()
            font.setPointSize(11)
            p.setFont(font)
            p.drawText(rect, Qt.AlignCenter | Qt.TextWordWrap, self._message)

        alarm = time.monotonic() < self._alarm_until
        if alarm:
            pulse = 0.5 + 0.5 * math.sin(time.monotonic() * 14)
            color = QtGui.QColor(DANGER)
            color.setAlphaF(0.18 + 0.22 * pulse)
            p.fillPath(clip, color)
            p.setPen(QtGui.QColor("#FFFFFF"))
            font = p.font()
            font.setPointSize(22)
            font.setBold(True)
            p.setFont(font)
            p.drawText(rect, Qt.AlignCenter | Qt.TextWordWrap, self._alarm_text)
        pen = QtGui.QPen(QtGui.QColor(DANGER if alarm else self._border), 3 if alarm else 1.5)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawPath(clip)
        p.end()


class Gauge(QtWidgets.QWidget):
    """Jauge circulaire de vigilance (0-100)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(190, 190)
        self._value = 100.0
        self._shown = 100.0
        self._color = OK
        self._active = False
        self._label = "En attente"
        self._label_color = MUTED

    def set_label(self, text: str, color: str = MUTED) -> None:
        self._label = text
        self._label_color = color
        self.update()

    def set_value(self, value: float, active: bool = True) -> None:
        self._value = max(0.0, min(100.0, value))
        self._active = active
        self._shown += (self._value - self._shown) * 0.25  # lissage visuel
        self._color = OK if self._shown >= 70 else WARN if self._shown >= 40 else DANGER
        self.update()

    def paintEvent(self, _event) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        side = min(self.width(), self.height()) - 24
        rect = QtCore.QRectF((self.width() - side) / 2, (self.height() - side) / 2, side, side)
        thickness = max(10.0, side * 0.075)

        p.setPen(QtGui.QPen(QtGui.QColor(BORDER), thickness, Qt.SolidLine, Qt.RoundCap))
        p.drawArc(rect, 225 * 16, -270 * 16)
        if self._active:
            p.setPen(QtGui.QPen(QtGui.QColor(self._color), thickness, Qt.SolidLine, Qt.RoundCap))
            p.drawArc(rect, 225 * 16, int(-270 * 16 * self._shown / 100.0))

        font = p.font()
        font.setPointSizeF(side * 0.2)
        font.setBold(True)
        p.setFont(font)
        p.setPen(QtGui.QColor(TEXT if self._active else MUTED))
        text = f"{self._shown:.0f}" if self._active else "—"
        p.drawText(rect.adjusted(0, -side * 0.12, 0, -side * 0.12), Qt.AlignCenter, text)
        font.setPointSizeF(max(9.0, side * 0.075))
        p.setFont(font)
        p.setPen(QtGui.QColor(self._label_color))
        p.drawText(rect.adjusted(0, side * 0.42, 0, 0), Qt.AlignHCenter | Qt.AlignVCenter,
                   self._label)
        p.end()


class Sparkline(QtWidgets.QWidget):
    """Courbes des 60 dernières secondes."""

    def __init__(self, series: list[tuple[str, str]], parent=None):
        super().__init__(parent)
        self.setMinimumHeight(110)
        self._series = series
        self._data = {name: deque(maxlen=300) for name, _ in series}

    def push(self, **values: float) -> None:
        for name, value in values.items():
            self._data[name].append(max(0.0, min(100.0, value)))
        self.update()

    def clear(self) -> None:
        for d in self._data.values():
            d.clear()
        self.update()

    def paintEvent(self, _event) -> None:
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        top, bottom = 6.0, h - 6.0
        p.setPen(QtGui.QPen(QtGui.QColor(BORDER), 1, Qt.DashLine))
        for frac in (0.0, 0.5, 1.0):
            y = top + (bottom - top) * frac
            p.drawLine(QtCore.QPointF(0, y), QtCore.QPointF(w, y))

        for i, (name, color) in enumerate(self._series):
            data = self._data[name]
            if len(data) < 2:
                continue
            n = data.maxlen
            step = w / (n - 1)
            x0 = w - (len(data) - 1) * step
            pts = [QtCore.QPointF(x0 + k * step, bottom - (bottom - top) * v / 100.0)
                   for k, v in enumerate(data)]
            path = QtGui.QPainterPath(pts[0])
            for pt in pts[1:]:
                path.lineTo(pt)
            if i == 0:  # aire dégradée sous la courbe principale
                area = QtGui.QPainterPath(path)
                area.lineTo(pts[-1].x(), bottom)
                area.lineTo(pts[0].x(), bottom)
                grad = QtGui.QLinearGradient(0, top, 0, bottom)
                c = QtGui.QColor(color)
                c.setAlpha(70)
                grad.setColorAt(0, c)
                c.setAlpha(0)
                grad.setColorAt(1, c)
                p.fillPath(area, grad)
            p.setPen(QtGui.QPen(QtGui.QColor(color), 2 if i == 0 else 1.4))
            p.drawPath(path)
        p.end()


class Tile(QtWidgets.QFrame):
    def __init__(self, title: str, hint: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(2)
        t = QtWidgets.QLabel(title.upper())
        t.setObjectName("cardTitle")
        self.value = QtWidgets.QLabel("—")
        self.value.setObjectName("tileValue")
        self.hint = QtWidgets.QLabel(hint)
        self.hint.setObjectName("tileHint")
        lay.addWidget(t)
        lay.addWidget(self.value)
        lay.addWidget(self.hint)

    def set(self, text: str, color: str = TEXT, hint: str | None = None) -> None:
        self.value.setText(text)
        self.value.setStyleSheet(f"color: {color};")
        if hint is not None:
            self.hint.setText(hint)


class HistoryDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Historique des sessions")
        self.resize(760, 420)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(18, 18, 18, 18)
        sessions = recent_sessions(50)
        headers = ["Date", "Profil", "Durée", "Vigilance moy.", "Vigilance min.",
                   "Distrait", "Bâillements", "Alarmes"]
        table = QtWidgets.QTableWidget(len(sessions), len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.verticalHeader().hide()
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        for r, s in enumerate(sessions):
            profile = PROFILES.get(s["profile"])
            cells = [
                datetime.fromtimestamp(s["started"]).strftime("%d/%m/%Y %H:%M"),
                profile.label if profile else s["profile"],
                _fmt_duration(s["duration_s"]),
                f"{s['avg_vigilance']:.0f} %",
                f"{s['min_vigilance']:.0f} %",
                _fmt_duration(s["distracted_s"]),
                str(s["yawns"]),
                str(s["alarms"]),
            ]
            for c, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(r, c, item)
        if not sessions:
            empty = QtWidgets.QLabel("Aucune session enregistrée pour l'instant.")
            empty.setObjectName("muted")
            lay.addWidget(empty)
        lay.addWidget(table)


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds or 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} h {m:02d}"
    if m:
        return f"{m} min {s:02d}"
    return f"{s} s"


# ------------------------------------------------------------- fenêtre principale
class MainWindow(QtWidgets.QMainWindow):
    frame_ready = QtCore.Signal()
    engine_error = QtCore.Signal(str)
    engine_status = QtCore.Signal(str)
    remote_command = QtCore.Signal(str)

    def __init__(self, profile: str = "bureau", camera: int = 0, sound: bool = True,
                 port: int = DEFAULT_PORT):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(make_icon())
        self.resize(1180, 700)
        self.setMinimumSize(980, 620)

        self._pending = None
        self._pending_alerts: list = []
        self._pending_lock = threading.Lock()
        self._scheduled = False
        self._last_plot = 0.0
        self._tray_state = None
        self._quitting = False
        self._tray_hint_shown = False

        self.engine = Engine(profile, camera, sound_enabled=sound,
                             on_update=self._on_engine_update,
                             on_error=self.engine_error.emit,
                             on_status=self.engine_status.emit)
        self.frame_ready.connect(self._consume, Qt.QueuedConnection)
        self.engine_error.connect(self._on_error)
        self.engine_status.connect(self._on_status)
        self.remote_command.connect(self._on_remote)

        self._build_ui(profile, camera, sound)
        self._build_tray()

        self.server = StatusServer(self.engine.status, port, on_command=self.remote_command.emit)
        if not self.server.start():
            self.footer.setText(f"Port {port} occupé : les extensions ne pourront pas se connecter.")

        self._watch = QtCore.QTimer(self)
        self._watch.timeout.connect(self._watchdog)
        self._watch.start(500)

    # ---------------------------------------------------------------- UI
    def _build_ui(self, profile: str, camera: int, sound: bool) -> None:
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(22, 18, 22, 14)
        outer.setSpacing(16)

        # En-tête
        header = QtWidgets.QHBoxLayout()
        header.setSpacing(10)
        logo = QtWidgets.QLabel()
        logo.setPixmap(make_icon().pixmap(36, 36))
        titles = QtWidgets.QVBoxLayout()
        titles.setSpacing(0)
        title = QtWidgets.QLabel(APP_NAME)
        title.setObjectName("title")
        self.subtitle = QtWidgets.QLabel("Détection de fatigue et de distraction")
        self.subtitle.setObjectName("subtitle")
        titles.addWidget(title)
        titles.addWidget(self.subtitle)
        header.addWidget(logo)
        header.addLayout(titles)
        header.addStretch(1)

        self.profile_box = QtWidgets.QComboBox()
        for key, p in PROFILES.items():
            self.profile_box.addItem(p.label, key)
            self.profile_box.setItemData(self.profile_box.count() - 1, p.description, Qt.ToolTipRole)
        self.profile_box.setCurrentIndex(max(0, self.profile_box.findData(profile)))
        self.profile_box.currentIndexChanged.connect(self._profile_changed)
        self.profile_box.setToolTip("Profil d'utilisation")

        self.camera_box = QtWidgets.QComboBox()
        for index in range(max(4, camera + 1)):
            self.camera_box.addItem(f"Caméra {index}", index)
        self.camera_box.setCurrentIndex(camera)
        self.camera_box.setToolTip("Caméra utilisée (0 = caméra intégrée)")

        self.sound_btn = QtWidgets.QPushButton("Son activé")
        self.sound_btn.setCheckable(True)
        self.sound_btn.setChecked(sound)
        self.sound_btn.toggled.connect(self._sound_toggled)
        self._sound_toggled(sound)

        calib_btn = QtWidgets.QPushButton("Recalibrer")
        calib_btn.setToolTip("Reprendre la calibration (5 s en regardant l'écran normalement)")
        calib_btn.clicked.connect(self.engine.recalibrate)
        history_btn = QtWidgets.QPushButton("Historique")
        history_btn.clicked.connect(lambda: HistoryDialog(self).exec())

        self.start_btn = QtWidgets.QPushButton("Démarrer")
        self.start_btn.setObjectName("primary")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.clicked.connect(self.toggle)

        for w in (self.profile_box, self.camera_box, self.sound_btn, calib_btn, history_btn,
                  self.start_btn):
            header.addWidget(w)
        outer.addLayout(header)

        self.profile_hint = QtWidgets.QLabel()
        self.profile_hint.setObjectName("muted")
        self.profile_hint.setAlignment(Qt.AlignRight)
        outer.addWidget(self.profile_hint)
        outer.setSpacing(10)

        # Corps
        body = QtWidgets.QHBoxLayout()
        body.setSpacing(16)
        outer.addLayout(body, 1)

        left = QtWidgets.QVBoxLayout()
        left.setSpacing(16)
        self.video = VideoView()
        left.addWidget(self.video, 1)

        chart = Card("Dernière minute")
        legend = QtWidgets.QHBoxLayout()
        for text, color in (("Vigilance", OK), ("Ouverture des yeux", VIOLET)):
            dot = QtWidgets.QLabel(f"<span style='color:{color}'>●</span> {text}")
            dot.setObjectName("muted")
            legend.addWidget(dot)
        legend.addStretch(1)
        chart.body.addLayout(legend)
        self.spark = Sparkline([("vigilance", OK), ("yeux", VIOLET)])
        chart.body.addWidget(self.spark)
        left.addWidget(chart)
        body.addLayout(left, 3)

        right = QtWidgets.QVBoxLayout()
        right.setSpacing(16)
        status = Card("Vigilance")
        self.gauge = Gauge()
        status.body.addWidget(self.gauge, 1)
        self.reason_label = QtWidgets.QLabel("La surveillance n'est pas démarrée")
        self.reason_label.setObjectName("muted")
        self.reason_label.setAlignment(Qt.AlignCenter)
        self.reason_label.setWordWrap(True)
        status.body.addWidget(self.reason_label)
        right.addWidget(status, 3)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(12)
        self.t_perclos = Tile("Yeux fermés", "du temps, sur 1 min")
        self.t_blinks = Tile("Clignements", "par minute")
        self.t_yawns = Tile("Bâillements", "sur 5 min")
        self.t_head = Tile("Tête", "orientation")
        self.t_distance = Tile("Distance", "à l'écran")
        self.t_session = Tile("Session", "durée")
        for i, tile in enumerate((self.t_perclos, self.t_blinks, self.t_yawns,
                                  self.t_head, self.t_distance, self.t_session)):
            grid.addWidget(tile, i // 3, i % 3)
        right.addLayout(grid)

        alerts = Card("Alertes")
        self.alert_list = QtWidgets.QListWidget()
        self.alert_list.setWordWrap(True)
        self.alert_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.alert_list.setFocusPolicy(Qt.NoFocus)
        alerts.body.addWidget(self.alert_list)
        right.addWidget(alerts, 2)
        body.addLayout(right, 2)

        self.footer = QtWidgets.QLabel(
            f"v{__version__} · Traitement 100 % local, aucune image enregistrée")
        self.footer.setObjectName("muted")
        outer.addWidget(self.footer)

    def _build_tray(self) -> None:
        self.tray = QtWidgets.QSystemTrayIcon(make_icon(MUTED), self)
        self.tray.setToolTip(APP_NAME)
        menu = QtWidgets.QMenu()
        menu.addAction("Afficher le tableau de bord", self.show_window)
        self.tray_toggle = menu.addAction("Démarrer la surveillance", self.toggle)
        menu.addSeparator()
        menu.addAction("Quitter", self.quit)
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(
            lambda reason: self.show_window()
            if reason in (QtWidgets.QSystemTrayIcon.Trigger, QtWidgets.QSystemTrayIcon.DoubleClick)
            else None)
        self.tray.show()

    # ----------------------------------------------------------- contrôle
    def show_window(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def toggle(self) -> None:
        if self.engine.running or self._starting():
            self.stop()
        else:
            self.start()

    def _starting(self) -> bool:
        t = self.engine._thread
        return bool(t and t.is_alive())

    def start(self) -> None:
        if self._starting():
            return
        self.engine.camera = self.camera_box.currentData()
        self.spark.clear()
        self.video.set_message("Démarrage…")
        self._set_running_ui(True)
        self.engine.start()

    def stop(self) -> None:
        self.engine.stop(wait=True)
        with self._pending_lock:
            self._pending = None
        self._set_running_ui(False)
        self.video.set_message("Surveillance arrêtée")
        self.video.set_border(BORDER)
        self.gauge.set_value(100, active=False)
        self.gauge.set_label("En attente")
        self.reason_label.setText("La surveillance n'est pas démarrée")
        self._set_tray(None)

    def _set_running_ui(self, running: bool) -> None:
        self.start_btn.setText("Arrêter" if running else "Démarrer")
        self.start_btn.setProperty("running", "true" if running else "false")
        self.start_btn.style().unpolish(self.start_btn)
        self.start_btn.style().polish(self.start_btn)
        self.camera_box.setEnabled(not running)
        self.tray_toggle.setText("Arrêter la surveillance" if running else "Démarrer la surveillance")

    def _profile_changed(self) -> None:
        key = self.profile_box.currentData()
        self.engine.set_profile(key)
        if not self._starting():
            self.engine.profile = PROFILES[key]
        self.profile_hint.setText(PROFILES[key].description)

    def _sound_toggled(self, on: bool) -> None:
        self.engine.sound_enabled = on
        self.sound_btn.setText("Son activé" if on else "Son coupé")

    def _on_remote(self, command: str) -> None:
        if command == "show":
            self.show_window()
        elif command == "start":
            self.start()
        elif command == "stop":
            self.stop()
        elif command == "recalibrate":
            self.engine.recalibrate()

    # ------------------------------------------------- flux moteur -> UI
    def _on_engine_update(self, frame, snap, face) -> None:
        """Appelé depuis le thread moteur : on ne garde que la dernière image."""
        with self._pending_lock:
            self._pending = (frame, snap, face)
            self._pending_alerts.extend(snap.alerts)
            if self._scheduled:
                return
            self._scheduled = True
        self.frame_ready.emit()

    def _consume(self) -> None:
        with self._pending_lock:
            item, self._pending = self._pending, None
            alerts, self._pending_alerts = self._pending_alerts, []
            self._scheduled = False
        if item is None:
            return
        frame, snap, face = item
        color = STATE_COLORS.get(snap.state, ACCENT)

        if self.isVisible() and not self.isMinimized():
            if face is not None:
                self._draw_overlay(frame, face, snap, color)
            self.video.set_border(color)
            self.video.set_frame(frame)

        calibrating = snap.state == CALIBRATION
        self.gauge.set_value(snap.vigilance, active=not calibrating and snap.state != ABSENT)
        self.gauge.set_label(snap.state_label, color)
        if calibrating:
            self.reason_label.setText(
                f"Regardez l'écran normalement… {snap.calibration * 100:.0f} %")
        elif snap.state == ATTENTIF:
            self.reason_label.setText("Tout va bien")
        else:
            self.reason_label.setText(snap.reason or snap.state_label)

        perclos = snap.perclos * 100
        self.t_perclos.set(f"{perclos:.0f} %", DANGER if perclos > 18 else WARN if perclos > 10 else TEXT)
        self.t_blinks.set(f"{snap.blink_rate:.0f}" if snap.blink_rate else "—")
        self.t_yawns.set(str(snap.yawns), WARN if snap.yawns >= 3 else TEXT)
        if snap.face_present:
            if snap.pitch < -HEAD_BACK_DEG:
                head = "En arrière"
            elif abs(snap.roll) > HEAD_TILT_DEG:
                head = "Penchée"
            elif snap.pitch > 15:
                head = "Baissée"
            elif abs(snap.yaw) > 20:
                head = "Tournée"
            else:
                head = "Droite"
            self.t_head.set(head, TEXT if head == "Droite" else WARN,
                            f"{snap.yaw:+.0f}° / {snap.pitch:+.0f}° / {snap.roll:+.0f}°")
            close = 0 < snap.distance_cm < TOO_CLOSE_CM
            self.t_distance.set(f"{snap.distance_cm:.0f} cm" if snap.distance_cm else "—",
                                WARN if close else TEXT)
        else:
            self.t_head.set("—", MUTED, "visage absent")
            self.t_distance.set("—", MUTED)
        self.t_session.set(_fmt_duration(snap.session_s))

        now = time.monotonic()
        if now - self._last_plot >= 0.2:
            self._last_plot = now
            self.spark.push(vigilance=snap.vigilance, yeux=snap.eye_open * 100)

        for alert in alerts:
            self._add_alert(alert)
        self._set_tray(snap.state)
        mode = "CNN + MediaPipe" if self.engine.cnn_active else "MediaPipe"
        self.footer.setText(
            f"v{__version__} · {self.engine.fps:.0f} images/s · {mode} · "
            "Traitement 100 % local, aucune image enregistrée")

    def _draw_overlay(self, frame, face, snap, color: str) -> None:
        h, w = frame.shape[:2]
        pts = np.empty((face.landmarks.shape[0], 2), dtype=np.int32)
        pts[:, 0] = face.landmarks[:, 0] * w
        pts[:, 1] = face.landmarks[:, 1] * h
        main = _bgr(color)
        dots = frame.copy()
        for x, y in pts[::2]:
            cv2.circle(dots, (int(x), int(y)), 1, main, -1, cv2.LINE_AA)
        cv2.addWeighted(dots, 0.35, frame, 0.65, 0, frame)
        cv2.polylines(frame, [pts[list(FACE_OVAL)]], True, main, 1, cv2.LINE_AA)
        eye_color = _bgr(DANGER) if snap.eyes_closed_s > 0.3 else _bgr(VIOLET)
        for loop in (LEFT_EYE_LOOP, RIGHT_EYE_LOOP):
            cv2.polylines(frame, [pts[list(loop)]], True, eye_color, 2, cv2.LINE_AA)
        mouth_color = _bgr(WARN) if snap.mouth_open > 0.55 else main
        cv2.polylines(frame, [pts[list(LIPS_LOOP)]], True, mouth_color, 1, cv2.LINE_AA)

    def _add_alert(self, alert) -> None:
        color = LEVEL_COLORS.get(alert.level, ACCENT)
        stamp = datetime.now().strftime("%H:%M:%S")
        item = QtWidgets.QListWidgetItem(f"{stamp}   {alert.message}")
        item.setForeground(QtGui.QColor(color))
        self.alert_list.insertItem(0, item)
        while self.alert_list.count() > 60:
            self.alert_list.takeItem(self.alert_list.count() - 1)

        if alert.level == "alarm":
            sleepy = alert.kind in ("microsleep", "drowsy", "sleep_posture", "sleep_lost")
            self.video.flash_alarm("RÉVEILLEZ-VOUS !" if sleepy else "ATTENTION !")
            QtWidgets.QApplication.alert(self, 3000)
        hidden = not self.isVisible() or self.isMinimized() or not self.isActiveWindow()
        if hidden and alert.level in ("warn", "alarm"):
            icon = (QtWidgets.QSystemTrayIcon.Critical if alert.level == "alarm"
                    else QtWidgets.QSystemTrayIcon.Warning)
            self.tray.showMessage(APP_NAME, alert.message, icon, 6000)

    def _set_tray(self, state: str | None) -> None:
        if state == self._tray_state:
            return
        self._tray_state = state
        color = STATE_COLORS.get(state, MUTED) if state else MUTED
        self.tray.setIcon(make_icon(color))
        label = self.engine.latest.state_label if state else "à l'arrêt"
        self.tray.setToolTip(f"{APP_NAME} — {label}")

    def _on_error(self, message: str) -> None:
        self._set_running_ui(False)
        self.video.set_message(message)
        self.gauge.set_label("Erreur", DANGER)
        self.reason_label.setText("La surveillance n'a pas pu démarrer")
        self.gauge.set_value(100, active=False)
        self._set_tray(None)

    def _on_status(self, message: str) -> None:
        if message:
            self.video.set_message(message)

    def _watchdog(self) -> None:
        # Anime le clignotement d'alarme même si aucune image n'arrive.
        if time.monotonic() < self.video._alarm_until:
            self.video.update()

    # -------------------------------------------------------------- fermeture
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._quitting or not self.tray.isVisible() or not self._starting():
            self.quit()
            event.accept()
            return
        # Surveillance en cours : on continue en arrière-plan.
        event.ignore()
        self.hide()
        if not self._tray_hint_shown:
            self._tray_hint_shown = True
            self.tray.showMessage(
                APP_NAME, "La surveillance continue en arrière-plan. "
                "Clic droit sur l'icône pour quitter.",
                QtWidgets.QSystemTrayIcon.Information, 5000)

    def quit(self) -> None:
        self._quitting = True
        self.engine.stop(wait=True)
        self.server.stop()
        self.tray.hide()
        QtWidgets.QApplication.quit()


def _register_windows_identity() -> None:
    """Donne à Windows le vrai nom de l'application.

    Sans cela, les notifications et la barre des tâches affichent « Python ».
    L'identité est déclarée pour l'utilisateur courant uniquement (HKCU).
    """
    if sys.platform != "win32":
        return
    app_id = "PhilippeCoder.FatigueGuard"
    try:
        import ctypes
        import winreg

        from .config import DATA_DIR

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        icon_path = DATA_DIR / "icon.png"
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER,
                              rf"Software\Classes\AppUserModelId\{app_id}") as key:
            winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, APP_NAME)
            winreg.SetValueEx(key, "IconUri", 0, winreg.REG_SZ, str(icon_path))
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass  # purement cosmétique : ne doit jamais empêcher le lancement


def run_gui(profile: str = "bureau", camera: int = 0, sound: bool = True,
            port: int = DEFAULT_PORT, autostart: bool = False, minimized: bool = False,
            use_cnn: bool = True) -> int:
    _register_windows_identity()
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    try:  # icône utilisée par Windows dans les notifications
        from .config import DATA_DIR

        make_icon(size=128).pixmap(128, 128).save(str(DATA_DIR / "icon.png"))
    except Exception:
        pass
    app.setQuitOnLastWindowClosed(False)
    app.setStyle("Fusion")
    app.setStyleSheet(QSS)
    window = MainWindow(profile, camera, sound, port)
    window.engine.use_cnn = use_cnn
    window._profile_changed()
    if minimized:
        autostart = True
    else:
        window.show()
    if autostart:
        QtCore.QTimer.singleShot(200, window.start)
    return app.exec()
