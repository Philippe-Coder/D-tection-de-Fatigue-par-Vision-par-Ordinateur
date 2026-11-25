"""Tests du serveur local, de l'historique et du moteur (sans caméra)."""
import json
import urllib.error
import urllib.request

import pytest

from fatigue_guard.analyzer import Alert, FatigueAnalyzer
from fatigue_guard.config import get_profile
from fatigue_guard.engine import Engine
from fatigue_guard.server import StatusServer
from fatigue_guard.storage import SessionRecorder, recent_sessions

PORT = 8791


@pytest.fixture
def server():
    received = []
    srv = StatusServer(lambda: {"running": True, "vigilance": 87.5}, PORT,
                       on_command=received.append)
    assert srv.start()
    yield srv, received
    srv.stop()


def _post(path, headers=None):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", method="POST",
                                 headers=headers or {})
    return urllib.request.urlopen(req, timeout=3)


def test_status_endpoint(server):
    data = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/status", timeout=3).read())
    assert data["app"] == "fatigue-guard"
    assert data["vigilance"] == 87.5


def test_commands_require_extension_header(server):
    _, received = server
    with pytest.raises(urllib.error.HTTPError) as err:
        _post("/stop")
    assert err.value.code == 403
    assert received == []
    assert _post("/stop", {"X-Fatigue-Guard": "1"}).status == 200
    assert received == ["stop"]


def test_cors_only_for_browser_extensions(server):
    def allow_origin(origin):
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/status", headers={"Origin": origin})
        return urllib.request.urlopen(req, timeout=3).headers.get("Access-Control-Allow-Origin")

    assert allow_origin("chrome-extension://abcdef") == "chrome-extension://abcdef"
    assert allow_origin("moz-extension://1234") == "moz-extension://1234"
    assert allow_origin("https://site-quelconque.example") is None

    preflight = urllib.request.Request(f"http://127.0.0.1:{PORT}/stop", method="OPTIONS",
                                       headers={"Origin": "https://site-quelconque.example"})
    response = urllib.request.urlopen(preflight, timeout=3)
    assert response.status == 204
    assert response.headers.get("Access-Control-Allow-Origin") is None


def test_unknown_command_is_rejected(server):
    with pytest.raises(urllib.error.HTTPError) as err:
        _post("/shutdown", {"X-Fatigue-Guard": "1"})
    assert err.value.code == 404


def test_second_server_on_same_port_fails_cleanly(server):
    other = StatusServer(dict, PORT)
    assert other.start() is False


def test_session_history_roundtrip(tmp_path):
    db = tmp_path / "history.db"
    analyzer = FatigueAnalyzer(get_profile("etude"))
    analyzer.total_yawns = 2
    analyzer.total_alarms = 1
    rec = SessionRecorder("etude", path=db)
    rec.add_events([Alert("yawn", "warn", "bâillement")])
    rec.close(analyzer, duration_s=125.0)
    sessions = recent_sessions(path=db)
    assert len(sessions) == 1
    assert sessions[0]["profile"] == "etude"
    assert sessions[0]["yawns"] == 2
    assert sessions[0]["duration_s"] == 125.0


def test_very_short_sessions_are_not_kept(tmp_path):
    db = tmp_path / "history.db"
    rec = SessionRecorder("bureau", path=db)
    rec.close(FatigueAnalyzer(get_profile("bureau")), duration_s=3.0)
    assert recent_sessions(path=db) == []


def test_engine_status_is_json_ready_before_start():
    status = Engine("conduite").status()
    assert status["running"] is False
    assert status["profile"] == "conduite"
    assert status["recent_alerts"] == []
    json.dumps(status)
