"""Single- and multi-camera CCTV processing pipelines."""
import cv2, csv, json, time, uuid
from pathlib import Path
from .detector import PersonDetector
from .tracker import IdentityTracker
from .global_identity import GlobalIdentityManager
from .features import OSNetExtractor
from .motion import estimate_camera_transform

def _make_reid_extractor(reid_extractor=None):
    """Return the configured deep person-ReID extractor.

    OSNet is deliberately the default for every production pipeline. Tests may
    inject a lightweight fake extractor to avoid loading model weights.
    """
    return reid_extractor if reid_extractor is not None else OSNetExtractor(device="cpu")


def _color(identity_id):
    seed=sum(ord(c) for c in str(identity_id))
    return ((seed*67)%200+40,(seed*131)%200+40,(seed*197)%200+40)

def _draw(frame, tracks, camera_id="CAM_01", global_manager=None, timestamp=0.0):
    out=frame.copy()
    for t in tracks:
        gid, status, score = (str(t.identity_id),"local",1.0)
        if global_manager is not None:
            gid,status,score=global_manager.resolve(camera_id,t.identity_id,t.appearance,timestamp,t.state)
        x1,y1,x2,y2=[int(v) for v in t.bbox]; color=_color(gid)
        cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
        label=f"{gid} | local {t.identity_id}"
        cv2.putText(out,label,(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,0.52,color,2)
    cv2.putText(out,f"ARGUS-X  {camera_id}  |  {len(tracks)} visible",(14,30),cv2.FONT_HERSHEY_SIMPLEX,0.75,(40,220,120),2)
    return out

def process_video(video_path, output_dir="outputs", model_path="yolov8n.pt", image_size=640,
                  confidence=0.30,max_lost_age=600,recovery_threshold=0.70,
                  camera_compensation=True,progress_callback=None,reid_extractor=None):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    sid=uuid.uuid4().hex[:12]; detector=PersonDetector(model_path,"cpu",image_size,confidence)
    reid_extractor = _make_reid_extractor(reid_extractor)
    tracker=IdentityTracker(max_lost_age=max_lost_age,recovery_threshold=recovery_threshold,
                            feature_extractor=reid_extractor)
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): raise RuntimeError(f"Cannot open video source: {video_path}")
    fps=cap.get(cv2.CAP_PROP_FPS); fps=fps if fps and fps>1 else 25.0
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    video=output_dir/f"{sid}_tracked.mp4"; csv_path=output_dir/f"{sid}_tracks.csv"; events_path=output_dir/f"{sid}_events.json"
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
    if not writer.isOpened(): cap.release(); raise RuntimeError("Could not create output video. Check codec and output folder permissions.")
    prev=None; frame_no=0; rows=[]; start=time.time()
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            frame_no+=1; gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            matrix=estimate_camera_transform(prev,gray) if camera_compensation else None
            visible=tracker.update(frame,detector.detect(frame),frame_no,matrix)
            writer.write(_draw(frame,visible))
            for t in visible:
                x1,y1,x2,y2=[float(v) for v in t.bbox]
                rows.append({"frame":frame_no,"camera_id":"CAM_01","local_track_id":t.identity_id,"global_identity_id":f"PERSON_{t.identity_id:04d}","x1":x1,"y1":y1,"x2":x2,"y2":y2,"confidence":t.confidence,"state":t.state})
            prev=gray
            if progress_callback and total: progress_callback(min(frame_no/total,1.0),frame_no,total)
    finally: cap.release(); writer.release()
    elapsed=max(time.time()-start,1e-6)
    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        fields=list(rows[0]) if rows else ["frame","camera_id","local_track_id","global_identity_id","x1","y1","x2","y2","confidence","state"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    with open(events_path,"w",encoding="utf-8") as f: json.dump(tracker.events,f,indent=2)
    return {"session_id":sid,"video":str(video),"tracks_csv":str(csv_path),"events_json":str(events_path),"frames":frame_no,"elapsed_seconds":elapsed,"fps":frame_no/elapsed,"summary":tracker.summary(),"tracker":tracker}

def process_multi_camera(video_sources, output_dir="outputs", model_path="yolov8n.pt", image_size=640,
                         confidence=0.30,max_lost_age=600,recovery_threshold=0.70,
                         cross_camera_threshold=0.78,cross_camera_margin=0.10,
                         cross_camera_window_seconds=30.0,progress_callback=None,reid_extractor=None):
    """Process uploaded videos in timestamp order and resolve shared global IDs.
    Videos are assumed to have approximately synchronized start times. This is an
    offline demonstration mode; real deployments need camera clock synchronization.
    """
    if len(video_sources)<2: raise ValueError("Multi-camera mode needs at least two video sources.")
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    sid=uuid.uuid4().hex[:12]; detector=PersonDetector(model_path,"cpu",image_size,confidence)
    reid_extractor = _make_reid_extractor(reid_extractor)
    manager=GlobalIdentityManager(cross_camera_threshold,cross_camera_margin,cross_camera_window_seconds)
    cams=[]
    for index,(camera_id,source) in enumerate(video_sources.items()):
        cap=cv2.VideoCapture(str(source))
        if not cap.isOpened():
            for c in cams: c["cap"].release()
            raise RuntimeError(f"Cannot open {camera_id}: {source}")
        fps=cap.get(cv2.CAP_PROP_FPS); fps=fps if fps and fps>1 else 25.0
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        path=output_dir/f"{sid}_{camera_id}_tracked.mp4"
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
        if not writer.isOpened(): cap.release(); raise RuntimeError(f"Could not create output for {camera_id}")
        cams.append({"id":camera_id,"cap":cap,"fps":fps,"total":total,"frame":0,"prev":None,"tracker":IdentityTracker(max_lost_age=max_lost_age,recovery_threshold=recovery_threshold,feature_extractor=reid_extractor),"writer":writer,"output":path,"rows":[],"done":False})
    start=time.time(); total_frames=sum(c["total"] for c in cams); processed=0
    try:
        # Round-robin is timestamp ordered for approximately synchronized files.
        while any(not c["done"] for c in cams):
            for c in cams:
                if c["done"]: continue
                ok,frame=c["cap"].read()
                if not ok: c["done"]=True; continue
                c["frame"]+=1; processed+=1; timestamp=(c["frame"]-1)/c["fps"]
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                matrix=estimate_camera_transform(c["prev"],gray)
                visible=c["tracker"].update(frame,detector.detect(frame),c["frame"],matrix)
                annotated=_draw(frame,visible,c["id"],manager,timestamp); c["writer"].write(annotated)
                for t in visible:
                    gid,status,score=manager.resolve(c["id"],t.identity_id,t.appearance,timestamp,t.state)
                    x1,y1,x2,y2=[float(v) for v in t.bbox]
                    c["rows"].append({"frame":c["frame"],"timestamp":timestamp,"camera_id":c["id"],"local_track_id":t.identity_id,"global_identity_id":gid,"identity_event":status,"match_confidence":score,"x1":x1,"y1":y1,"x2":x2,"y2":y2})
                c["prev"]=gray
                if progress_callback and total_frames: progress_callback(min(processed/total_frames,1.0),processed,total_frames)
    finally:
        for c in cams: c["cap"].release(); c["writer"].release()
    elapsed=max(time.time()-start,1e-6); outputs=[]
    for c in cams:
        csv_path=output_dir/f"{sid}_{c['id']}_tracks.csv"
        with open(csv_path,"w",newline="",encoding="utf-8") as f:
            fields=list(c["rows"][0]) if c["rows"] else ["frame","timestamp","camera_id","local_track_id","global_identity_id","identity_event","match_confidence","x1","y1","x2","y2"]
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(c["rows"])
        outputs.append({"camera_id":c["id"],"video":str(c["output"]),"tracks_csv":str(csv_path),"frames":c["frame"],"summary":c["tracker"].summary()})
    events_path=output_dir/f"{sid}_global_events.json"
    with open(events_path,"w",encoding="utf-8") as f: json.dump(manager.events,f,indent=2)
    return {"session_id":sid,"cameras":outputs,"global_identities":len(manager.identities),"global_events":manager.events,"events_json":str(events_path),"elapsed_seconds":elapsed,"fps":processed/elapsed}


def process_live_stream(source, duration_seconds=30, output_dir="outputs", model_path="yolov8n.pt",
                        image_size=416, confidence=0.30, max_lost_age=600,
                        recovery_threshold=0.70, progress_callback=None, reid_extractor=None):
    """Run live person tracking on one webcam index or RTSP/HTTP CCTV stream.

    The stream is sampled until duration_seconds elapses. Some RTSP sources buffer;
    actual wall-clock duration can therefore vary by camera/network.
    """
    from .live import open_camera_source
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    sid=uuid.uuid4().hex[:12]; detector=PersonDetector(model_path,"cpu",image_size,confidence)
    reid_extractor = _make_reid_extractor(reid_extractor)
    tracker=IdentityTracker(max_lost_age=max_lost_age,recovery_threshold=recovery_threshold,feature_extractor=reid_extractor)
    cap=open_camera_source(source)
    fps=cap.get(cv2.CAP_PROP_FPS); fps=fps if fps and fps>1 else 20.0
    ok,first=cap.read()
    if not ok: cap.release(); raise RuntimeError("Camera opened but did not provide a frame.")
    height,width=first.shape[:2]; video=output_dir/f"{sid}_live_tracked.mp4"
    csv_path=output_dir/f"{sid}_live_tracks.csv"; events_path=output_dir/f"{sid}_live_events.json"
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
    if not writer.isOpened(): cap.release(); raise RuntimeError("Could not create live output video.")
    start=time.time(); frame_no=0; rows=[]; frame=first
    try:
        while time.time()-start < float(duration_seconds):
            frame_no+=1; detections=detector.detect(frame)
            visible=tracker.update(frame,detections,frame_no)
            writer.write(_draw(frame,visible,"LIVE_CAM"))
            for t in visible:
                x1,y1,x2,y2=[float(v) for v in t.bbox]
                rows.append({"frame":frame_no,"camera_id":"LIVE_CAM","local_track_id":t.identity_id,"global_identity_id":f"PERSON_{t.identity_id:04d}","x1":x1,"y1":y1,"x2":x2,"y2":y2,"confidence":t.confidence,"state":t.state})
            if progress_callback: progress_callback(min((time.time()-start)/duration_seconds,1.0),frame_no,0)
            ok,frame=cap.read()
            if not ok: break
    finally: cap.release(); writer.release()
    elapsed=max(time.time()-start,1e-6)
    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        fields=list(rows[0]) if rows else ["frame","camera_id","local_track_id","global_identity_id","x1","y1","x2","y2","confidence","state"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    with open(events_path,"w",encoding="utf-8") as f: json.dump(tracker.events,f,indent=2)
    return {"session_id":sid,"video":str(video),"tracks_csv":str(csv_path),"events_json":str(events_path),"frames":frame_no,"elapsed_seconds":elapsed,"fps":frame_no/elapsed,"summary":tracker.summary(),"tracker":tracker}
