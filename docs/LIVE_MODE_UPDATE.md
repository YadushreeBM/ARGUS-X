# ARGUS-X live-mode update

## What changed
- Replaced the old fixed-duration, record-then-display live tab with a WebRTC live video component.
- The browser captures the selected webcam and sends frames to the local Streamlit server.
- Each frame passes through YOLO person detection, the local identity tracker, and the global identity registry.
- The annotated frame is returned immediately with both `PERSON_####` global IDs and `LOCAL-###` local IDs.
- Live processor counters and recent global identity events are exposed in the dashboard.

## Run
```powershell
pip install -r requirements.txt
streamlit run app.py
```
Open the local app, select the Live CCTV / IP Stream tab, click START in the video component, and allow camera permission.

## Important scope
This update provides live browser-webcam processing for a locally running dashboard. It does not yet provide live RTSP/NVR ingestion or synchronized live multi-camera fusion. The global registry uses the current HSV appearance descriptor and is not a trained deep Re-ID model. A visible ID overlay is not evidence of correct identity recovery; validate with controlled footage containing occlusion and re-entry.
