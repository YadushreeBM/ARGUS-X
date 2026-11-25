"""Central configuration for ARGUS-X."""
from dataclasses import dataclass

@dataclass
class AppConfig:
    model_path: str = "yolov8n.pt"
    device: str = "cpu"
    image_size: int = 640
    detection_confidence: float = 0.30
    max_active_age: int = 30
    max_lost_age: int = 600
    recovery_threshold: float = 0.70
    recovery_margin: float = 0.10
    min_hits: int = 2
    camera_compensation: bool = True
    max_detections: int = 100
    cross_camera_threshold: float = 0.78
    cross_camera_margin: float = 0.10
    cross_camera_window_seconds: float = 30.0
