# Suggested 5-minute judging presentation

1. Problem: object identity is often lost after occlusion, camera movement and track fragmentation.
2. Existing baseline: detection + online association can produce fragmented identities.
3. Proposed ARGUS-X: persistent identity memory and confidence-aware recovery.
4. Architecture: detector, motion, association, appearance, memory, recovery and event logging.
5. Live demo: show a track disappearing and returning; show an ambiguous candidate being rejected.
6. Evaluation: baseline vs ARGUS-X using IDF1, ID switches, recovery precision, false recovery rate and FPS.
7. Limitations and next steps: stronger deep ReID embeddings, class-specific features, multi-camera synchronization and calibrated confidence.

Only present features that have been implemented and tested in the submitted build.
