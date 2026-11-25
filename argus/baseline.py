"""Ultralytics ByteTrack baseline for fair comparison with ARGUS-X."""
import cv2
import csv
from pathlib import Path
from ultralytics import YOLO

def run_bytetrack_baseline(video_path, output_dir="outputs", model_path="yolov8n.pt",
                           image_size=416, confidence=0.30):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = output_dir / "bytetrack_baseline.mp4"
    output_csv = output_dir / "bytetrack_baseline.csv"
    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    rows = []
    frame_no = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            result = model.track(
                frame, persist=True, tracker="bytetrack.yaml",
                imgsz=image_size, conf=confidence, device="cpu",
                verbose=False
            )[0]
            annotated = result.plot()
            writer.write(annotated)
            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                for box, identity, cls, score in zip(boxes, ids, classes, confs):
                    rows.append({
                        "frame": frame_no, "identity_id": int(identity),
                        "class_id": int(cls), "x1": float(box[0]),
                        "y1": float(box[1]), "x2": float(box[2]),
                        "y2": float(box[3]), "confidence": float(score)
                    })
    finally:
        cap.release()
        writer.release()
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fields = ["frame","identity_id","class_id","x1","y1","x2","y2","confidence"]
        out = csv.DictWriter(f, fieldnames=fields)
        out.writeheader()
        out.writerows(rows)
    return {"video": str(output_video), "tracks_csv": str(output_csv),
            "frames": frame_no, "detections": len(rows)}
