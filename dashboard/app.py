"""Flask + SocketIO dashboard server."""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename

import config


class DashboardServer:
    """Serve live dashboard APIs, MJPEG stream, and websocket updates."""

    def __init__(self, event_logger, frame_queue, shared_state: dict | None = None):
        self.logger = logging.getLogger(__name__)
        self.event_logger = event_logger
        self.frame_queue = frame_queue
        self.shared_state = shared_state or {"confidence_threshold": config.CONFIDENCE_THRESHOLD, "alert_mute": False}
        self.shared_state.setdefault("video_source", None)
        self.shared_state.setdefault("camera_active", True)  # False = live cam paused
        self.state_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.latest_frame: bytes | None = None
        self.upload_dir = config.BASE_DIR / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.allowed_video_exts = {"mp4", "avi", "mov", "mkv"}

        self.app = Flask(
            __name__,
            template_folder=str((config.BASE_DIR / "dashboard/templates")),
            static_folder=str((config.BASE_DIR / "dashboard/static")),
        )
        self.app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        self._register_routes()
        threading.Thread(target=self._frame_ingest_loop, daemon=True).start()

    def _register_routes(self) -> None:
        @self.app.get("/")
        def index():
            return render_template("index.html", camera_zone=config.CAMERA_ZONE)

        @self.app.get("/api/events")
        def api_events():
            return jsonify(self.event_logger.get_recent(50))

        @self.app.get("/api/stats")
        def api_stats():
            return jsonify(self.event_logger.get_stats())

        @self.app.get("/api/last_alert")
        def api_last_alert():
            """Return the most recent alert event for persistence across page refresh."""
            rows = self.event_logger.get_recent(1)
            if rows:
                return jsonify({"ok": True, "event": rows[0]})
            return jsonify({"ok": False, "event": None})

        @self.app.post("/api/config")
        def api_config():
            data = request.get_json(silent=True) or {}
            with self.state_lock:
                if "confidence_threshold" in data:
                    self.shared_state["confidence_threshold"] = float(data["confidence_threshold"])
                if "alert_mute" in data:
                    self.shared_state["alert_mute"] = bool(data["alert_mute"])
            return jsonify({"ok": True, "config": self.shared_state})

        @self.app.post("/api/video/upload")
        def api_video_upload():
            if "file" not in request.files:
                return jsonify({"ok": False, "error": "Missing file"}), 400
            file = request.files["file"]
            if not file or not file.filename:
                return jsonify({"ok": False, "error": "No file selected"}), 400

            ext = Path(file.filename).suffix.lower().lstrip(".")
            if ext not in self.allowed_video_exts:
                return jsonify({"ok": False, "error": "Unsupported file type"}), 400

            base_name = secure_filename(Path(file.filename).stem) or "video"
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"{base_name}_{timestamp}.{ext}"
            path = self.upload_dir / filename
            file.save(str(path))

            with self.state_lock:
                self.shared_state["video_source"] = str(path)

            return jsonify({"ok": True, "filename": filename})

        @self.app.post("/api/video/stop")
        def api_video_stop():
            with self.state_lock:
                self.shared_state["video_source"] = None
            # Notify clients to switch back to live camera UI
            self.socketio.emit("video_stopped", {})
            return jsonify({"ok": True})

        @self.app.post("/api/camera/stop")
        def api_camera_stop():
            """Pause live camera processing and show a frozen/blank frame."""
            with self.state_lock:
                self.shared_state["camera_active"] = False
            return jsonify({"ok": True, "camera_active": False})

        @self.app.post("/api/camera/start")
        def api_camera_start():
            """Resume live camera processing."""
            with self.state_lock:
                self.shared_state["camera_active"] = True
            return jsonify({"ok": True, "camera_active": True})

        @self.app.get("/video_feed")
        def video_feed():
            return Response(self._frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

    def _placeholder_frame(self) -> bytes:
        frame = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
        cv2.putText(frame, "No feed", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 136), 3)
        ok, encoded = cv2.imencode(".jpg", frame)
        return encoded.tobytes() if ok else b""

    def _frame_generator(self):
        while True:
            with self.frame_lock:
                payload = self.latest_frame
            if payload is None:
                payload = self._placeholder_frame()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"

    def _frame_ingest_loop(self) -> None:
        while True:
            try:
                frame = self.frame_queue.get(timeout=1.0)
            except Empty:
                continue
            if isinstance(frame, bytes):
                payload = frame
            else:
                ok, encoded = cv2.imencode(".jpg", frame)
                payload = encoded.tobytes() if ok else self._placeholder_frame()
            with self.frame_lock:
                self.latest_frame = payload

    def emit_new_alert(self, payload: dict) -> None:
        self.socketio.emit("new_alert", payload)

    def emit_detection_state(self, payload: dict | None) -> None:
        """Emit current detection state; pass None to reset dashboard to idle."""
        if payload is None:
            self.socketio.emit("detection_reset", {})
        else:
            self.socketio.emit("detection_state", payload)

    def emit_frame_stats(self, payload: dict) -> None:
        self.socketio.emit("frame_stats", payload)

    def run(self) -> None:
        self.socketio.run(self.app, host="0.0.0.0", port=config.FLASK_PORT, debug=config.FLASK_DEBUG, use_reloader=False)


if __name__ == "__main__":
    from queue import Queue

    from logger.event_logger import EventLogger

    logging.basicConfig(level=logging.INFO)
    server = DashboardServer(EventLogger(str(config.DB_PATH)), Queue(maxsize=5))
    logging.getLogger(__name__).info(
        "Dashboard self-test: routes ready %s", sorted([r.rule for r in server.app.url_map.iter_rules()])
    )
