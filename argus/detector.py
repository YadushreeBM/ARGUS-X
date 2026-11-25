"""CPU-first YOLO person detector. COCO class 0 is person."""
from ultralytics import YOLO

class PersonDetector:
    def __init__(self, model_path="yolov8n.pt", device="cpu", image_size=640,
                 confidence=0.30, max_detections=100):
        self.model = YOLO(model_path)
        self.device = device
        self.image_size = image_size
        self.confidence = confidence
        self.max_detections = max_detections

    def detect(self, frame):
        results = self.model.predict(source=frame, imgsz=self.image_size,
            conf=self.confidence, device=self.device, max_det=self.max_detections,
            classes=[0], verbose=False)
        detections = []
        if not results or results[0].boxes is None:
            return detections
        for box in results[0].boxes:
            detections.append({
                "bbox": box.xyxy[0].cpu().numpy().astype(float).tolist(),
                "confidence": float(box.conf[0].cpu().item()),
                "class_id": 0, "class_name": "person"
            })
        return detections

# Backwards-compatible import name for existing project code.
ObjectDetector = PersonDetector
