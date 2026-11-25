# ARGUS-X — Track the Untrackable

**Solo-built, CPU-first CCTV person tracking and identity recovery prototype.**

ARGUS-X detects people in authorized CCTV footage, assigns camera-local track IDs, maintains appearance and motion history, attempts to recover lost identities, and supports an offline multi-camera demonstration that maps local tracks to shared global person IDs.

## Features
- YOLO person-only detection (COCO class `person`, class 0).
- Per-camera tracking with Hungarian data association.
- HSV appearance descriptors and bounded feature memory.
- Lost-track memory and confidence/margin-gated recovery.
- Sparse optical-flow camera-motion compensation.
- Offline multi-camera global identity association.
- Streamlit dashboard for recorded single-camera and multi-camera videos.
- Real-time browser webcam processing through WebRTC with live person boxes, camera-local IDs and global identity registry IDs.
- SQLite session, identity and event persistence.
- Annotated MP4, CSV and JSON outputs.

## Run in VS Code (Windows)
1. Extract the ZIP.
2. Open the `ARGUS-X` folder in VS Code.
3. Open Terminal → New Terminal.
4. Run:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again. Python 3.10–3.12 is recommended. First launch downloads `yolov8n.pt` automatically; internet is needed for that first download.

## Dashboard modes
- **Single CCTV · Recorded:** upload one video and process local person tracking and recovery.
- **Multi-CCTV · Recorded:** upload two or more approximately time-synchronized recordings. Each is processed with its own local tracker; the global identity manager attempts appearance-based cross-camera association.
- **Live CCTV / Webcam:** browser webcam frames are processed continuously through WebRTC and returned with live person boxes, Local IDs and Global IDs. This mode uses the browser's selected camera and requires camera permission. RTSP/IP stream live ingestion and live multi-camera fusion are not included in this release.

## CLI
Single camera:

```powershell
python main.py --video sample.mp4
```

Multi-camera offline demo:

```powershell
python multi_camera_main.py --camera CAM_01=cam1.mp4 --camera CAM_02=cam2.mp4
```

## Outputs
Generated in `outputs/`: annotated videos, tracking CSVs and identity event JSON. SQLite database: `argus_x.db`.

## Live mode notes
- The live dashboard uses `streamlit-webrtc`; click START in the video component and allow browser camera access.
- Local IDs come from the per-camera tracker. Global IDs are assigned by the global identity registry; in a single-camera session this is a session-scoped global registry, not cross-camera proof.
- The current appearance descriptor remains HSV histograms, not a trained deep person-ReID model. Identity recovery is probabilistic and must be validated on real sequences.

## Important limitations (be honest in evaluation)
- HSV histograms are lightweight appearance descriptors, not a trained deep person-ReID model. Similar clothing, lighting changes, pose and camera viewpoints can cause errors.
- Cross-camera matching is appearance-led and uses approximate synchronized timestamps. Real deployments need synchronized camera clocks, camera topology/travel-time constraints and stronger Re-ID embeddings.
- Global identity IDs are session-scoped in this prototype; database persistence stores session summaries/events, not a complete restartable live identity-feature gallery.
- Live stream mode currently previews a frame; continuous live inference and live multi-camera fusion are planned extensions, not claimed as complete.
- Camera-motion compensation uses sparse optical flow and can fail with low texture, heavy parallax or large scene changes.
- YOLOv8n detects only classes it was trained for; this build intentionally filters to people.
- No identity match is guaranteed. The system should be evaluated on representative sequences and uncertain matches should not be forced.

## Tests
```powershell
python -m pytest -q
```

## Responsible use
Use only CCTV footage and camera streams you are authorized to access. ARGUS-X is a research/hackathon prototype, not a certified security or biometric identification product.
