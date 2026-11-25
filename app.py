"""ARGUS-X professional CCTV identity tracking dashboard."""
import streamlit as st, pandas as pd, tempfile, os, cv2
from pathlib import Path
from argus.pipeline import process_video, process_multi_camera, process_live_stream
from argus.storage import initialize, save_session, save_multi_session

st.set_page_config(page_title="ARGUS-X | CCTV Identity Intelligence",page_icon="🎯",layout="wide")
initialize()
st.markdown("""<style>
:root{color-scheme:dark}.stApp{background:#080f1b}.block-container{padding-top:1.4rem;max-width:1500px}
[data-testid="stMetric"]{background:#111d2d;padding:14px;border:1px solid #23354d;border-radius:12px}
.argus-sub{color:#9fb2ca;font-size:1rem}.panel{background:#111d2d;border:1px solid #263950;border-radius:14px;padding:16px}
</style>""",unsafe_allow_html=True)
st.title("🎯 ARGUS-X")
st.markdown('<p class="argus-sub">TRACK THE UNTRACKABLE · Long-Horizon Person Identity Recovery</p>',unsafe_allow_html=True)
st.caption("CCTV surveillance analytics · Person-only detection · Persistent identity memory · Confidence-aware recovery")

with st.sidebar:
    st.header("Engine configuration")
    model=st.text_input("YOLO model weights","yolov8n.pt",help="Downloads automatically on first run if not present.")
    imgsz=st.select_slider("Inference resolution",options=[320,416,512,640],value=416)
    conf=st.slider("Person detection confidence",0.10,0.90,0.30,0.05)
    lost=st.slider("Lost identity memory (frames)",30,1800,600,30)
    recover=st.slider("Single-camera recovery threshold",0.40,0.95,0.70,0.01)
    st.divider(); st.caption("CPU mode is enabled by default. Processing speed depends on your laptop and video resolution.")

tab1,tab2,tab3=st.tabs(["Single CCTV · Recorded","Multi-CCTV · Recorded","Live CCTV · Webcam"])

def save_upload(upload):
    suffix=Path(upload.name).suffix or ".mp4"
    tmp=tempfile.NamedTemporaryFile(delete=False,suffix=suffix)
    tmp.write(upload.getbuffer()); tmp.close(); return tmp.name

def render_single(result):
    s=result["summary"]; a,b,c,d=st.columns(4)
    a.metric("Frames processed",result["frames"]); b.metric("Processing FPS",f"{result['fps']:.2f}")
    c.metric("Unique person IDs",s["unique_identities"]); d.metric("Recovered IDs",s["recovered"])
    st.subheader("Annotated CCTV output")
    if Path(result["video"]).exists():
        st.video(result["video"])
        with open(result["video"],"rb") as f: st.download_button("Download annotated video",f,"argus_x_single_camera.mp4")
    if Path(result["events_json"]).exists():
        events=pd.read_json(result["events_json"])
        st.subheader("Identity event timeline"); st.dataframe(events,use_container_width=True)
        with open(result["events_json"],"rb") as f: st.download_button("Download event log JSON",f,"argus_x_events.json")
    with open(result["tracks_csv"],"rb") as f: st.download_button("Download tracking CSV",f,"argus_x_tracks.csv")

with tab1:
    st.subheader("Single-camera identity continuity")
    st.write("Upload one CCTV recording. ARGUS-X detects people, assigns IDs, remembers lost tracks and attempts to recover the original identity when a person reappears.")
    up=st.file_uploader("Select CCTV recording",type=["mp4","avi","mov","mkv"],key="single_upload")
    if up and st.button("▶ Analyze CCTV footage",type="primary",use_container_width=True):
        path=save_upload(up); prog=st.progress(0.0); status=st.empty()
        try:
            with st.spinner("Initializing CPU detection and identity engine..."):
                res=process_video(path,model_path=model,image_size=imgsz,confidence=conf,max_lost_age=lost,recovery_threshold=recover,progress_callback=lambda v,n,t:(prog.progress(v),status.write(f"Processing frame {n} / {t}")))
            save_session(res["session_id"],up.name,res["frames"],res["fps"],res["tracker"])
            st.session_state["single_result"]=res; st.success("CCTV analysis completed.")
        except Exception as e: st.error(f"Processing failed: {e}")
        finally:
            try: os.unlink(path)
            except OSError: pass
    if st.session_state.get("single_result"): render_single(st.session_state["single_result"])

with tab2:
    st.subheader("Cross-camera global identity recovery")
    st.warning("For this offline demo, recordings should start at approximately the same time and have synchronized timestamps. Appearance-only cross-camera matching is conservative and can be confused by similar clothing.")
    uploads=st.file_uploader("Select two or more camera recordings",type=["mp4","avi","mov","mkv"],accept_multiple_files=True,key="multi_upload")
    if uploads and st.button("▶ Analyze multiple CCTV cameras",type="primary",use_container_width=True):
        paths={}; prog=st.progress(0.0); status=st.empty()
        try:
            for i,u in enumerate(uploads): paths[f"CAM_{i+1:02d}"]=save_upload(u)
            with st.spinner("Processing camera streams and resolving global identities..."):
                res=process_multi_camera(paths,model_path=model,image_size=imgsz,confidence=conf,max_lost_age=lost,recovery_threshold=recover,progress_callback=lambda v,n,t:(prog.progress(v),status.write(f"Frames processed across cameras: {n} / {t}")))
            save_multi_session(res,"; ".join(u.name for u in uploads)); st.session_state["multi_result"]=res; st.success("Multi-camera analysis completed.")
        except Exception as e: st.error(f"Multi-camera processing failed: {e}")
        finally:
            for p in paths.values():
                try: os.unlink(p)
                except OSError: pass
    mr=st.session_state.get("multi_result")
    if mr:
        a,b,c=st.columns(3); a.metric("Cameras processed",len(mr["cameras"])); b.metric("Global identities",mr["global_identities"]); c.metric("Cross-camera recovery events",sum(e["event"]=="cross_camera_recovered" for e in mr["global_events"]))
        for camera in mr["cameras"]:
            with st.expander(f"{camera['camera_id']} · {camera['frames']} frames",expanded=True):
                st.video(camera["video"])
                with open(camera["video"],"rb") as f: st.download_button(f"Download {camera['camera_id']} annotated video",f,f"{camera['camera_id']}_argus_x.mp4",key=camera["camera_id"]+"vid")
                with open(camera["tracks_csv"],"rb") as f: st.download_button(f"Download {camera['camera_id']} tracking CSV",f,f"{camera['camera_id']}_tracks.csv",key=camera["camera_id"]+"csv")
        st.subheader("Global identity events"); st.dataframe(pd.DataFrame(mr["global_events"]),use_container_width=True)
        with open(mr["events_json"],"rb") as f: st.download_button("Download global event log",f,"argus_x_global_events.json")

with tab3:
    st.subheader("Real-time local CCTV / Webcam monitoring")
    st.info(
        "This live mode uses OpenCV to open the webcam connected to the computer "
        "running ARGUS-X. It does not use WebRTC, so model startup and camera "
        "processing cannot block the rest of the dashboard."
    )
    st.warning(
        "Allow camera access in Windows. Close Teams/Zoom/Camera app before starting. "
        "For this local desktop setup, start with camera index 0."
    )

    camera_index = st.number_input(
        "Camera device index", min_value=0, max_value=5, value=0, step=1,
        help="Usually 0 for the built-in laptop webcam; try 1 for an external webcam."
    )
    camera_id = st.text_input("Camera label", value="CAM_01", key="live_camera_label")

    if "live_camera_session" not in st.session_state:
        st.session_state["live_camera_session"] = None

    start_col, info_col = st.columns([1, 2])
    with start_col:
        start_live = st.button(
            "▶ Start live identity tracking", type="primary",
            use_container_width=True, key="start_live_camera"
        )
    with info_col:
        st.caption("Live overlay format: GLOBAL PERSON ID | LOCAL TRACK ID")

    if start_live:
        old_session = st.session_state.get("live_camera_session")
        if old_session is not None:
            try:
                old_session.close()
            except Exception:
                pass
        try:
            from argus.live_camera import LiveCameraSession
            with st.spinner("Opening camera and loading YOLO + OSNet Re-ID models..."):
                st.session_state["live_camera_session"] = LiveCameraSession(
                    camera_index=int(camera_index),
                    model_path=model,
                    image_size=imgsz,
                    confidence=conf,
                    max_lost_age=lost,
                    recovery_threshold=recover,
                    camera_id=camera_id.strip() or "CAM_01",
                )
            st.success("Live camera and identity engine initialized.")
        except Exception as exc:
            st.session_state["live_camera_session"] = None
            st.error(f"Live camera could not start: {type(exc).__name__}: {exc}")

    def render_live_camera():
        session = st.session_state.get("live_camera_session")
        if session is None:
            st.caption("The dashboard is ready. Press Start when you want to initialize the live camera.")
            return

        stop = st.button("■ Stop live camera", key="stop_live_camera")
        if stop:
            try:
                session.close()
            finally:
                st.session_state["live_camera_session"] = None
            st.info("Live camera stopped.")
            return

        frame, stats = session.read_and_process()
        if frame is not None:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        if stats.get("error"):
            st.error(stats["error"])

        a, b, c, d = st.columns(4)
        a.metric("Frames processed", stats.get("frames", 0))
        b.metric("Processing FPS", stats.get("fps", 0.0))
        c.metric("People visible now", stats.get("visible_people", 0))
        d.metric("Recovered local tracks", stats.get("recovered_local_ids", 0))

        left, right = st.columns(2)
        with left:
            st.subheader("Current Local IDs")
            st.write(stats.get("local_ids") or "No people detected in the current frame.")
        with right:
            st.subheader("Current Global IDs")
            st.write(stats.get("global_ids") or "No global identities in the current frame.")

        st.caption(
            f"Unique local tracks: {stats.get('unique_local_ids', 0)} · "
            f"Persistent global identities: {stats.get('unique_global_ids', 0)}"
        )
        st.subheader("Recent global identity events")
        events = stats.get("events", [])
        if events:
            st.dataframe(pd.DataFrame(events), use_container_width=True)
        else:
            st.caption("Identity creation and recovery events will appear here.")

    # Fragment refreshes only the live panel. The rest of the dashboard remains
    # available and is not reinitialized by each camera frame.
    if st.session_state.get("live_camera_session") is not None and hasattr(st, "fragment"):
        @st.fragment(run_every="0.15s")
        def live_camera_fragment():
            render_live_camera()
        live_camera_fragment()
    else:
        render_live_camera()

st.divider()
st.caption("ARGUS-X is a research/hackathon prototype. Identity recovery is probabilistic, not guaranteed. Use authorized CCTV footage and validate performance on representative test sequences.")
