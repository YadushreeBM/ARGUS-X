"""Real-time browser webcam processing for the ARGUS-X Streamlit dashboard.

The browser captures the webcam through WebRTC. Each incoming frame is processed
by YOLO -> local identity tracker -> global identity registry, then returned
immediately with Local ID and Global ID overlays.
"""
from __future__ import annotations

import threading
import time
import cv2
import numpy as np
from av import VideoFrame
from streamlit_webrtc import VideoProcessorBase

from .detector import PersonDetector
from .tracker import IdentityTracker
from .global_identity import GlobalIdentityManager
from .features import OSNetExtractor


class LiveIdentityProcessor(VideoProcessorBase):
    def __init__(
        self,
        model_path="yolov8n.pt",
        image_size=416,
        confidence=0.30,
        max_lost_age=600,
        recovery_threshold=0.70,
        camera_id="CAM_01",
        reid_extractor=None,
    ):
        self.detector = PersonDetector(
            model_path=model_path,
            device="cpu",
            image_size=image_size,
            confidence=confidence,
        )
        # The Streamlit app preloads and caches this model before WebRTC starts.
        # This avoids model download/initialization inside the WebRTC worker.
        self.reid = reid_extractor or OSNetExtractor(
            device="cpu", model_name="osnet_x0_25"
        )
        self.tracker = IdentityTracker(
            max_active_age=30,
            max_lost_age=max_lost_age,
            recovery_threshold=recovery_threshold,
            feature_extractor=self.reid,
        )
        self.global_manager = GlobalIdentityManager(
            threshold=0.72,
            margin=0.08,
            max_gap_seconds=30.0,
        )
        self.camera_id = str(camera_id)
        self.frame_index = 0
        self.started_at = time.time()
        self.last_error = None
        self._lock = threading.Lock()
        self._stats = {
            "frames": 0,
            "fps": 0.0,
            "visible_people": 0,
            "local_ids": [],
            "global_ids": [],
            "events": [],
            "error": None,
        }

    @staticmethod
    def _color(global_id: str):
        seed = sum(ord(ch) for ch in str(global_id))
        return (
            (seed * 67) % 190 + 50,
            (seed * 131) % 190 + 50,
            (seed * 197) % 190 + 50,
        )

    def recv(self, frame: VideoFrame) -> VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        self.frame_index += 1
        now = time.time()

        try:
            detections = self.detector.detect(image)
            visible = self.tracker.update(image, detections, self.frame_index)
            local_ids, global_ids = [], []

            for track in visible:
                global_id, status, score = self.global_manager.resolve(
                    self.camera_id,
                    track.identity_id,
                    track.appearance,
                    now,
                    track.state,
                )
                local_ids.append(int(track.identity_id))
                global_ids.append(global_id)

                x1, y1, x2, y2 = [int(v) for v in track.bbox]
                color = self._color(global_id)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                label = f"{global_id} | LOCAL-{track.identity_id:03d}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2
                )
                top = max(0, y1 - th - 10)
                cv2.rectangle(image, (x1, top), (x1 + tw + 8, top + th + 8), color, -1)
                cv2.putText(
                    image, label, (x1 + 4, top + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2,
                    cv2.LINE_AA,
                )

            elapsed = max(now - self.started_at, 1e-6)
            fps = self.frame_index / elapsed
            cv2.rectangle(image, (0, 0), (image.shape[1], 42), (12, 25, 39), -1)
            cv2.putText(
                image,
                f"ARGUS-X LIVE  |  {self.camera_id}  |  PEOPLE: {len(visible)}  |  {fps:.1f} FPS",
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (80, 240, 160), 2,
                cv2.LINE_AA,
            )

            with self._lock:
                self._stats.update({
                    "frames": self.frame_index,
                    "fps": round(fps, 2),
                    "visible_people": len(visible),
                    "local_ids": local_ids,
                    "global_ids": global_ids,
                    "events": list(self.global_manager.events[-20:]),
                    "error": None,
                })
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._stats["error"] = self.last_error
            cv2.putText(
                image, "ARGUS-X PROCESSING ERROR", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
            )

        return VideoFrame.from_ndarray(image, format="bgr24")

    def snapshot(self):
        with self._lock:
            result = dict(self._stats)
            result["local_ids"] = list(self._stats["local_ids"])
            result["global_ids"] = list(self._stats["global_ids"])
            result["events"] = list(self._stats["events"])
            result["tracker_summary"] = self.tracker.summary()
            return result
