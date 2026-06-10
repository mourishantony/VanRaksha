"""VanRaksha AI entrypoint."""
from __future__ import annotations

import logging
import math
import queue
import threading
import time
from datetime import datetime, timezone

import cv2

import config
from dashboard.app import DashboardServer
from detection.classifier import SpeciesClassifier
from detection.detector import WildlifeDetector
from logger.event_logger import EventLogger
from output.led_controller import LEDController
from output.sms_alert import SMSAlert
from output.voice import VoiceAlert
from threat.alert_level import AlertLevel, get_alert_level
from threat.scorer import ThreatScorer
from threat.species_index import SPECIES_CLASS_NAMES, get_risk_label
from utils.drawing import annotate_detection


# Label normalisation: maps raw YOLO output names → VanRaksha species keys.
# Combined model (wildlife_combined.pt) outputs: tiger, leopard, bear, deer
# COCO model     (yolov8n.pt)            outputs: elephant, person
LABEL_ALIASES = {
    # Human variants
    "person":   "human",
    "people":   "human",
    # Bear → maps to species_index key (which has 'sloth_bear' for risk label,
    # but we also accept plain 'bear' from the new trained model)
    "bear":     "bear",   # handled directly — risk label: HIGH RISK – SLOTH BEAR
}

# COCO class IDs that correspond to human body parts / accessories.
# These are frequently misclassified as animals by a custom tiger model.
_COCO_BODY_PART_CLASS_IDS = {
    # 'ear' doesn't exist in COCO but some custom models extend COCO with
    # person-parts; block anything the tiger model calls < class 0 safety.
    # More importantly, block these COCO person-adjacent IDs:
    0,    # person (handled separately as human; not a wild animal)
}

# Minimum number of consecutive frames a species must appear before it is
# treated as a confirmed detection (temporal smoothing).
_CONFIRM_FRAMES = 3

# Minimum detector confidence to pass temporal smoothing (stricter than
# the runtime threshold which can be lowered by the user).
_MIN_SMOOTH_CONF = 0.45


def _normalize_label(label: str) -> str:
    value = (label or "").strip().lower()
    return LABEL_ALIASES.get(value, value)


def _bbox_center(bbox: list) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _closest_human_distance(animal_bbox: list, human_detections: list[dict]) -> float:
    """Return the pixel distance to the nearest human, or -1.0 if none present."""
    if not human_detections:
        return -1.0
    ax, ay = _bbox_center(animal_bbox)
    return min(
        math.hypot(ax - _bbox_center(det["bbox"])[0], ay - _bbox_center(det["bbox"])[1])
        for det in human_detections
    )


def _human_near_animal(animal_bbox: list, human_detections: list[dict], threshold_px: float) -> bool:
    """Return True if the nearest human is within threshold_px of the animal."""
    dist = _closest_human_distance(animal_bbox, human_detections)
    return 0.0 <= dist <= threshold_px


def _is_camera_source(source: int | str) -> bool:
    if isinstance(source, int):
        return True
    if isinstance(source, str) and source.isdigit():
        return True
    return False


def _open_capture(source: int | str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    if _is_camera_source(source):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap


def main():
    """Run camera ingestion, detection, alerts, and dashboard updates."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("main")

    frame_queue = queue.Queue(maxsize=5)
    shared_state = {"confidence_threshold": config.CONFIDENCE_THRESHOLD, "alert_mute": False, "video_source": None}

    event_logger = EventLogger(str(config.DB_PATH))

    # ── Dual detector setup ──────────────────────────────────────────────────
    # Tiger detector: custom model trained specifically on tigers.
    tiger_detector = WildlifeDetector(str(config.TIGER_MODEL_PATH), config.CONFIDENCE_THRESHOLD)
    log.info("Tiger model loaded: %s", config.TIGER_MODEL_PATH)

    # General detector: COCO model for person + elephant detection.
    general_detector = WildlifeDetector(str(config.YOLO_MODEL_PATH), config.CONFIDENCE_THRESHOLD)
    log.info("General model loaded: %s", config.YOLO_MODEL_PATH)
    # ─────────────────────────────────────────────────────────────────────────

    classifier = SpeciesClassifier(str(config.CLASSIFIER_MODEL_PATH), SPECIES_CLASS_NAMES)
    scorer = ThreatScorer()
    voice = VoiceAlert(language=config.TTS_LANGUAGE, use_online=config.USE_ONLINE_TTS, cooldown_seconds=config.VOICE_COOLDOWN_SECONDS)
    led = LEDController(mode=config.LED_MODE)
    sms = SMSAlert(provider=config.SMS_PROVIDER)

    dashboard = DashboardServer(event_logger, frame_queue, shared_state)
    threading.Thread(target=dashboard.run, daemon=True).start()

    active_source: int | str = config.CAMERA_SOURCE
    cap = _open_capture(active_source)
    if not cap.isOpened():
        log.warning("Input source unavailable; dashboard will show No feed")

    # Wildlife targets from combined model (tiger/leopard/bear/deer) + COCO (elephant)
    wild_labels = {"tiger", "leopard", "bear", "deer", "elephant"}
    human_labels = {_normalize_label(label) for label in config.HUMAN_LABELS if label}
    allowed_labels = wild_labels | human_labels

    # ── Temporal smoothing state ─────────────────────────────────────────────
    # Maps species → consecutive-frame count at or above confidence threshold.
    _species_streak: dict[str, int] = {}
    # Tracks whether each species is currently in "confirmed" state so we
    # only log/count a new event when the detection first becomes confirmed
    # (not every frame thereafter).
    _species_confirmed: dict[str, bool] = {}
    # ─────────────────────────────────────────────────────────────────────────

    fps_counter = 0
    tick = time.time()
    try:
        while True:
            desired_source = shared_state.get("video_source") or config.CAMERA_SOURCE
            if desired_source != active_source:
                cap.release()
                active_source = desired_source
                cap = _open_capture(active_source)

            if not cap.isOpened():
                if shared_state.get("video_source"):
                    log.warning("Video source unavailable; reverting to camera")
                    shared_state["video_source"] = None
                    active_source = config.CAMERA_SOURCE
                    cap.release()
                    cap = _open_capture(active_source)
                if not cap.isOpened():
                    time.sleep(0.5)
                    dashboard.emit_frame_stats({"fps": 0, "detections": 0})
                    continue

            ok, frame = cap.read()
            if not ok:
                if not _is_camera_source(active_source):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if not ok:
                        log.warning("Video ended or unreadable; reverting to camera")
                        shared_state["video_source"] = None
                        active_source = config.CAMERA_SOURCE
                        cap.release()
                        cap = _open_capture(active_source)
                        continue
                else:
                    continue

            # ── Camera paused? Push placeholder and skip detection ──────────
            if not shared_state.get("camera_active", True):
                pause_frame = frame.copy()
                cv2.rectangle(pause_frame, (0, 0), (pause_frame.shape[1] - 1, pause_frame.shape[0] - 1), (60, 60, 60), 8)
                cv2.putText(pause_frame, "CAMERA PAUSED", (pause_frame.shape[1] // 2 - 160, pause_frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 80, 255), 3)
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_queue.put_nowait(pause_frame)
                time.sleep(0.05)
                continue
            # ────────────────────────────────────────────────────────────────

            conf_threshold = float(shared_state.get("confidence_threshold", config.CONFIDENCE_THRESHOLD))
            tiger_detector.confidence_threshold = conf_threshold
            general_detector.confidence_threshold = conf_threshold

            # ── Run both detectors and merge results ─────────────────────────
            tiger_detections = tiger_detector.detect(frame)   # outputs "tiger"
            general_detections = general_detector.detect(frame)  # outputs COCO labels
            raw_detections = tiger_detections + general_detections
            # ────────────────────────────────────────────────────────────────

            filtered_detections = []
            for det in raw_detections:
                label_norm = _normalize_label(det.get("label", "unknown"))
                if label_norm not in allowed_labels:
                    continue
                det["label_norm"] = label_norm
                filtered_detections.append(det)

            human_detections = [det for det in filtered_detections if det["label_norm"] in human_labels]
            if wild_labels:
                animal_detections = [det for det in filtered_detections if det["label_norm"] in wild_labels]
            else:
                animal_detections = [det for det in filtered_detections if det["label_norm"] not in human_labels]
            distance_threshold_px = max(1.0, config.HUMAN_ANIMAL_DISTANCE_RATIO * frame.shape[1])

            for det in human_detections:
                bbox = det["bbox"]
                label = f"human {det['confidence']:.2f}"
                annotate_detection(
                    frame,
                    bbox,
                    "human",
                    det["confidence"],
                    0.0,
                    AlertLevel.CAUTION,
                    draw_frame_border=False,
                    label_override=label,
                )

            # Species seen this frame (for streak tracking)
            species_this_frame: set[str] = set()

            for det in animal_detections:
                bbox = det["bbox"]
                yolo_label = det.get("label_norm", det.get("label", "unknown"))

                # ── False-positive guard: skip COCO body-part class IDs ──────
                if det.get("class_id") in _COCO_BODY_PART_CLASS_IDS and yolo_label not in wild_labels:
                    continue
                # Block tiny bboxes from the tiger model that are likely noise
                # (e.g. ear patches): require bbox area ≥ 0.5% of frame area.
                frame_area = frame.shape[0] * frame.shape[1]
                bbox_area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
                if bbox_area < 0.005 * frame_area:
                    continue
                # ─────────────────────────────────────────────────────────────

                cls = classifier.classify(frame, bbox, yolo_label=yolo_label)
                species = _normalize_label(cls["species"])

                # ── Temporal smoothing ────────────────────────────────────────
                species_this_frame.add(species)
                if det["confidence"] >= conf_threshold:
                    _species_streak[species] = _species_streak.get(species, 0) + 1
                else:
                    _species_streak[species] = 0

                confirmed = _species_streak.get(species, 0) >= _CONFIRM_FRAMES
                first_confirmation = confirmed and not _species_confirmed.get(species, False)
                if confirmed:
                    _species_confirmed[species] = True
                # ─────────────────────────────────────────────────────────────

                # Compute pixel distance to nearest human (used for distance-band scoring).
                distance_px = _closest_human_distance(bbox, human_detections)

                score = scorer.score(
                    species,
                    bbox,
                    frame.shape,
                    len(animal_detections),
                    det["confidence"],
                    distance_px=distance_px if distance_px >= 0 else None,
                )
                level = get_alert_level(score)

                # CRITICAL override: animal is within the proximity threshold of a human.
                if 0.0 <= distance_px <= distance_threshold_px:
                    level = AlertLevel.CRITICAL
                    score = max(score, 90.0)

                # Resolve human-readable risk label for this species.
                risk_label = get_risk_label(species)

                annotate_detection(frame, bbox, species, det["confidence"], score, level, risk_label=risk_label)

                # Always update live dashboard panel (even before confirmation)
                # so the operator can see what is being tracked.
                dashboard.emit_detection_state(
                    {
                        "species": species,
                        "confidence": det["confidence"],
                        "score": score,
                        "alert_level": level.name,
                        "risk_label": risk_label,
                        "confirmed": confirmed,
                    }
                )

                # Only act (alert / log / count) on the first confirmation frame
                # to prevent duplicate counting across continuous detections.
                if not first_confirmation:
                    continue

                voice_played = False
                if level in (AlertLevel.CAUTION, AlertLevel.HIGH, AlertLevel.CRITICAL) and not shared_state.get("alert_mute", False):
                    voice_played = voice.speak(species, level, "nearby", camera_zone=config.CAMERA_ZONE)

                sms_sent = False
                if level in (AlertLevel.HIGH, AlertLevel.CRITICAL):
                    led.set_alert(level)
                    sms_sent = sms.send(species, level, config.CAMERA_ZONE, score)
                    dashboard.emit_new_alert(
                        {
                            "species": species,
                            "confidence": det["confidence"],
                            "score": score,
                            "alert_level": level.name,
                            "risk_label": risk_label,
                        }
                    )

                event_logger.log_event(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "camera_zone": config.CAMERA_ZONE,
                        "species": species,
                        "confidence": det["confidence"],
                        "threat_score": score,
                        "alert_level": level.name,
                        "risk_label": risk_label,
                        "bbox": bbox,
                        "sms_sent": sms_sent,
                        "voice_played": voice_played,
                    }
                )

                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] {species.upper()} | {risk_label} | "
                    f"conf:{det['confidence']:.2f} | score:{score:.1f} | {level.name} | "
                    f"SMS:{'sent' if sms_sent else 'skip'} | Voice:{'played' if voice_played else 'skip'}"
                )

            # ── Reset streak for species not seen this frame ───────────────
            for sp in list(_species_streak.keys()):
                if sp not in species_this_frame:
                    _species_streak[sp] = 0
                    _species_confirmed[sp] = False

            # ── Emit reset when no animals are detected ────────────────────
            if not animal_detections:
                dashboard.emit_detection_state(None)

            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass
            frame_queue.put_nowait(frame)

            fps_counter += 1
            now = time.time()
            if now - tick >= 1:
                dashboard.emit_frame_stats({"fps": fps_counter, "detections": len(animal_detections)})
                fps_counter = 0
                tick = now

    except KeyboardInterrupt:
        log.info("Shutting down VanRaksha AI")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
