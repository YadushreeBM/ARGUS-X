# Evaluation plan

Use authorized, labeled sequences. Keep a fixed test set and record configuration/version for every run.

## Scenarios
- Continuous walking with direction changes.
- Short and prolonged partial/full occlusion.
- Exit and re-entry at different image positions.
- Similar-looking people crossing paths.
- Camera pan/tilt or mild shake.
- Two synchronized camera views of the same people.
- Non-overlapping camera views with known plausible transition times.

## Metrics
- Detection: precision, recall and mAP where annotations permit.
- Tracking: HOTA, IDF1, MOTA and ID switches using a valid MOT evaluation protocol.
- Recovery: recovery precision/recall, false recovery rate and recovery latency.
- Runtime: end-to-end FPS, CPU/RAM usage.

Compare custom ARGUS-X recovery against a ByteTrack-only baseline on the same videos and detector settings. Do not treat the number of recovery events as an accuracy metric. Report failed/ambiguous cases as well as successful cases.

## Dataset notes
MOT17/MOT20 are person-tracking datasets; VisDrone contains multiple object categories but this build filters to people. Check dataset terms and annotation formats before downloading. Start with small validation/test subsets to stay within the project storage limit.
