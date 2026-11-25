"""CLI: python multi_camera_main.py --camera CAM_01=one.mp4 --camera CAM_02=two.mp4"""
import argparse
from argus.pipeline import process_multi_camera
from argus.storage import save_multi_session

def main():
    p=argparse.ArgumentParser(); p.add_argument("--camera",action="append",required=True,help="Repeat CAM_ID=video_path")
    p.add_argument("--model",default="yolov8n.pt"); p.add_argument("--imgsz",type=int,default=416)
    a=p.parse_args(); sources={}
    for item in a.camera:
        if "=" not in item: p.error("Camera arguments must be CAM_ID=video_path")
        k,v=item.split("=",1); sources[k]=v
    r=process_multi_camera(sources,model_path=a.model,image_size=a.imgsz); save_multi_session(r)
    print({k:v for k,v in r.items() if k!="global_events"})
if __name__=="__main__": main()
