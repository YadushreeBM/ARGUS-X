"""Direct local-camera live pipeline for ARGUS-X.

This deliberately avoids streamlit-webrtc. The Streamlit process opens the
local webcam with OpenCV, processes one frame per UI fragment refresh, and
keeps one tracker + global identity registry for the complete live session.
"""
from __future__ import annotations

import time
import cv2
import numpy as np

from .detector import PersonDetector
from .features import OSNetExtractor
from .tracker import IdentityTracker
from .global_identity import GlobalIdentityManager


class LiveCameraSession:
    def __init__(
        self,
        camera_index=0,
        model_path="yolov8n.pt",
        image_size=416,
        confidence=0.30,
        max_lost_age=600,
        recovery_threshold=0.70,
        camera_id="CAM_01",
        reid_extractor=None,
    ):
        self.camera_id = str(camera_id).strip() or "CAM_01"
        self.camera_index = int(camera_index)
        self.cap = self._open_camera(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open webcam index {self.camera_index}. "
                "Close other camera applications, check Windows camera permission, "
                "or try camera index 1."
            )

        try:
            self.detector = PersonDetector(
                model_path=model_path,
                device="cpu",
                image_size=int(image_size),
                confidence=float(confidence),
            )
            self.reid = reid_extractor or OSNetExtractor(
                device="cpu", model_name="osnet_x0_25"
            )
            self.tracker = IdentityTracker(
                max_active_age=30,
                max_lost_age=int(max_lost_age),
                recovery_threshold=float(recovery_threshold),
                recovery_margin=0.10,
                feature_extractor=self.reid,
            )
            # The global registry outlives individual local track IDs. This is
            # what lets a newly-created local track be compared to old identities.
            self.global_manager = GlobalIdentityManager(
                threshold=0.72,
                margin=0.08,
                max_gap_seconds=max(120.0, float(max_lost_age) / 10.0),
            )
        except Exception:
            self.cap.release()
            raise

        self.frame_index = 0
        self.started_at = time.monotonic()
        self.closed = False
        self.last_stats = {
            "frames": 0, "fps": 0.0, "visible_people": 0,
            "local_ids": [], "global_ids": [], "events": [],
            "recovered_local_ids": 0, "error": None,
        }

    @staticmethod
    def _open_camera(index):
        # CAP_DSHOW avoids slow backend probing on many Windows laptops.
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    @staticmethod
    def _color(identity_id):
        seed = sum(ord(ch) for ch in str(identity_id))
        return ((seed * 67) % 190 + 50,
                (seed * 131) % 190 + 50,
                (seed * 197) % 190 + 50)

    def read_and_process(self):
        if self.closed:
            return None, dict(self.last_stats)
        ok, image = self.cap.read()
        if not ok or image is None:
            self.last_stats["error"] = "Webcam frame read failed. Check camera connection/permission."
            return None, dict(self.last_stats)

        self.frame_index += 1
        timestamp = time.monotonic() - self.started_at
        try:
            detections = self.detector.detect(image)
            visible = self.tracker.update(image, detections, self.frame_index)
            local_ids, global_ids = [], []

            for track in visible:
                global_id, status, score = self.global_manager.resolve(
                    self.camera_id,
                    track.identity_id,
                    track.appearance,
                    timestamp,
                    track.state,
                )
                local_ids.append(int(track.identity_id))
                global_ids.append(global_id)

                x1, y1, x2, y2 = [int(v) for v in track.bbox]
                color = self._color(global_id)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                label = f"{global_id} | LOCAL-{track.identity_id:03d}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                top = max(0, y1 - th - 10)
                cv2.rectangle(image, (x1, top), (x1 + tw + 8, top + th + 8), color, -1)
                cv2.putText(image, label, (x1 + 4, top + th + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2,
                            cv2.LINE_AA)

            elapsed = max(time.monotonic() - self.started_at, 1e-6)
            fps = self.frame_index / elapsed
            cv2.rectangle(image, (0, 0), (image.shape[1], 42), (12, 25, 39), -1)
            cv2.putText(image,
                        f"ARGUS-X LIVE | {self.camera_id} | PEOPLE: {len(visible)} | {fps:.1f} FPS",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (80, 240, 160), 2, cv2.LINE_AA)

            summary = self.tracker.summary()
            self.last_stats = {
                "frames": self.frame_index,
                "fps": round(fps, 2),
                "visible_people": len(visible),
                "local_ids": local_ids,
                "global_ids": global_ids,
                "events": list(self.global_manager.events[-20:]),
                "recovered_local_ids": summary["recovered"],
                "unique_local_ids": summary["unique_identities"],
                "unique_global_ids": len(self.global_manager.identities),
                "error": None,
            }
            return image, dict(self.last_stats)
        except Exception as exc:
            self.last_stats["error"] = f"{type(exc).__name__}: {exc}"
            cv2.putText(image, "ARGUS-X PROCESSING ERROR", (15, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return image, dict(self.last_stats)

    def close(self):
        if not self.closed:
            self.closed = True
            self.cap.release()
