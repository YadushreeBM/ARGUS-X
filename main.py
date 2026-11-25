"""CLI for single-camera ARGUS-X processing."""
import argparse
from argus.pipeline import process_video
from argus.storage import save_session

def main():
    p=argparse.ArgumentParser(description="ARGUS-X person identity tracking")
    p.add_argument("--video",required=True); p.add_argument("--model",default="yolov8n.pt")
    p.add_argument("--imgsz",type=int,default=416); p.add_argument("--conf",type=float,default=.30)
    p.add_argument("--max-lost",type=int,default=600); p.add_argument("--recovery-threshold",type=float,default=.70)
    p.add_argument("--no-camera-comp",action="store_true"); a=p.parse_args()
    r=process_video(a.video,model_path=a.model,image_size=a.imgsz,confidence=a.conf,max_lost_age=a.max_lost,recovery_threshold=a.recovery_threshold,camera_compensation=not a.no_camera_comp)
    save_session(r["session_id"],a.video,r["frames"],r["fps"],r["tracker"])
    print("ARGUS-X complete:", {k:v for k,v in r.items() if k!="tracker"})
if __name__=="__main__": main()
