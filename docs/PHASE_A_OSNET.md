# ARGUS-X Phase A — OSNet Appearance Re-ID

## What changed
- Live WebRTC processing now initializes OSNet x0.25 through Torchreid on CPU, using the official Market-1501 person-ReID checkpoint (not the ImageNet initialization).
- Person crops are resized and normalized for OSNet; output embeddings are L2-normalized.
- Local lost-track recovery and global identity matching consume the same deep embeddings.
- Similarity uses cosine similarity for deep vectors; the HSV Bhattacharyya comparison remains available for legacy/offline paths.
- Local tracks receive a short missed-detection grace period before being marked lost.

## Installation (Windows / PowerShell)
Run in the extracted ARGUS-X project folder:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Torch/Torchvision installation can vary by Python version and CPU platform. If pip reports a Torch/Torchvision wheel error, install a matching CPU build from the official PyTorch installation selector, then rerun the requirements command.

On first app start, ARGUS-X downloads the official OSNet x0.25 Market-1501 person-ReID checkpoint from the Torchreid Model Zoo into `weights/osnet_x0_25_market1501.pth.tar`. Internet access is required only for this first download. The Streamlit app caches the loaded model and initializes it before WebRTC starts, rather than downloading weights inside the WebRTC worker. Keep the checkpoint file for later offline use.

## Run

```powershell
streamlit run app.py
```

Open **Live CCTV · Webcam**, press START, and allow browser camera access.

## Important validation limits
- This is a Phase-A integration, not a claim of proven re-identification accuracy.
- OSNet embeddings describe visual appearance; they do not identify a real-world person or guarantee identity.
- Similar clothing, occlusion, low resolution, camera angle, lighting, and partial-body crops can cause false matches or missed matches.
- Thresholds (local recovery 0.70 by default; global 0.72 with 0.08 margin) are initial engineering values and must be calibrated on representative CCTV clips.
- The live processor currently extracts an embedding for each visible person on each processed frame. CPU FPS may be lower; benchmark on the target laptop before the hackathon demo.
- Phase B (motion + temporal + camera context fusion) is intentionally not implemented yet.

## Acceptance tests before Phase B
1. One person continuously visible for 60 seconds: measure local ID switches.
2. Person leaves and re-enters: measure whether the old identity is recovered.
3. Two people in similar clothing: measure false merges and identity swaps.
4. Repeat under different lighting and camera distances.
5. Report ID switches, recovery precision/recall, false-match count, and CPU FPS. Do not report synthetic unit tests as CCTV accuracy.


## Checkpoint provenance
The selected checkpoint is the Torchreid Model Zoo `osnet_x0_25` same-domain Market-1501 model. The code builds the matching 751-class architecture with `pretrained=False` and loads this ReID checkpoint explicitly. This is distinct from `build_model(..., pretrained=True)`, which loads ImageNet initialization. Model-zoo benchmark results are not a guarantee of performance on the user's CCTV cameras.

## Pipeline integration requirement

All recorded single-camera, recorded multi-camera, and sampled live-stream
pipelines must pass an `OSNetExtractor` instance into `IdentityTracker` as
`feature_extractor`. Without this dependency injection, the tracker falls
back to HSV colour histograms and the Market-1501 OSNet checkpoint is not used
for identity recovery. The pipeline now creates one CPU OSNet extractor by
default and shares it across cameras in a multi-camera run. Tests or controlled
experiments can inject a compatible extractor through the `reid_extractor`
argument.
