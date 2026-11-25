"""Run the standard YOLO + ByteTrack baseline."""
import argparse
from argus.baseline import run_bytetrack_baseline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.30)
    args = parser.parse_args()
    print(run_bytetrack_baseline(args.video, model_path=args.model,
                                 image_size=args.imgsz, confidence=args.conf))
