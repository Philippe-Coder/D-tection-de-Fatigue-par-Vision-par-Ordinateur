"""Petit serveur d'état local, lu par les extensions (VS Code, navigateur).

Il n'écoute que sur 127.0.0.1 et n'expose que des chiffres : jamais d'image.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import __version__
from .config import DEFAULT_PORT

COMMANDS = ("start", "stop", "show", "recalibrate")
EXTENSION_ORIGINS = ("chrome-extension://", "moz-extension://")


class _Server(ThreadingHTTPServer):
    # Par défaut, Windows laisse deux programmes écouter sur le même port :
    # on l'interdit pour détecter proprement une instance déjà lancée.
    allow_reuse_address = False
    daemon_threads = True


class StatusServer:
    def __init__(self, get_status, port: int = DEFAULT_PORT, on_command=None):
        self._get_status = get_status
        self._on_command = on_command
        self.port = port
        self._httpd: _Server | None = None

    def start(self) -> bool:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code: int, payload: dict) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self._cors()
                self.end_headers()
                self.wfile.write(body)

            def _cors(self) -> None:
                # Seules les extensions de navigateur sont autorisées en
                # cross-origin ; les pages web ordinaires restent bloquées.
                origin = self.headers.get("Origin", "")
                if origin.startswith(EXTENSION_ORIGINS):
                    self.send_header("Access-Control-Allow-Origin", origin)
                    self.send_header("Access-Control-Allow-Headers", "X-Fatigue-Guard")
                    self.send_header("Access-Control-Allow-Methods", "GET, POST")
                    self.send_header("Vary", "Origin")

            def do_OPTIONS(self):  # noqa: N802
                self.send_response(204)
                self._cors()
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self):  # noqa: N802
                if self.path.split("?")[0] in ("/", "/status"):
                    status = {"app": "fatigue-guard", "version": __version__}
                    status.update(outer._get_status())
                    self._send(200, status)
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):  # noqa: N802
                command = self.path.strip("/")
                # En-tête personnalisé obligatoire : une page web quelconque ne peut
                # pas l'envoyer vers 127.0.0.1, seules nos extensions le peuvent.
                if self.headers.get("X-Fatigue-Guard") != "1":
                    self._send(403, {"error": "forbidden"})
                elif outer._on_command and command in COMMANDS:
                    outer._on_command(command)
                    self._send(200, {"ok": True})
                else:
                    self._send(404, {"error": "unknown command"})

            def log_message(self, *args):  # silence
                pass

        try:
            self._httpd = _Server(("127.0.0.1", self.port), Handler)
        except OSError:
            return False  # port déjà pris (autre instance)
        threading.Thread(target=self._httpd.serve_forever, name="status-server",
                         daemon=True).start()
        return True

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
