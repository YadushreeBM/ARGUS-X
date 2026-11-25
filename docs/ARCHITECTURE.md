# Architecture

1. **Input layer:** recorded video files; live camera/IP URL preview.
2. **Detection:** YOLOv8n, restricted to COCO person class 0.
3. **Camera-local tracking:** Hungarian assignment using IoU, predicted center and HSV appearance; ByteTrack remains a baseline to compare against, not the custom identity engine.
4. **Identity memory:** local track records retain appearance features, velocity, last-seen frame and bounded feature history.
5. **Recovery:** lost tracks are matched using appearance-led scores with temporal/motion support, an absolute threshold and best-vs-second-best margin.
6. **Camera motion:** sparse optical flow estimates an affine transform for bounding-box prediction.
7. **Global identity:** offline multi-camera manager maps camera-local tracks to session-level `PERSON_####` IDs using appearance similarity, time window and ambiguity checks.
8. **Persistence/UI:** SQLite stores sessions and events; Streamlit exposes recorded processing and outputs.

## Identity namespace
A camera-local key is `(camera_id, local_track_id)`. A global ID is `PERSON_####`. Local IDs may differ between cameras while global IDs can be shared if a cross-camera match passes gates.

## Multi-camera synchronization assumption
The current offline demo interleaves video frames using each file's FPS and assumes approximate common start time. Unsynchronized recordings can produce invalid temporal comparisons. Production integration needs timestamps from camera/NVR metadata or an explicit synchronization step.
