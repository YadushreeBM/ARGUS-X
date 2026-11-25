"""Download/load OSNet ReID weights and run a one-crop inference smoke test.

Run from the ARGUS-X project root:
    python tools/verify_osnet.py
This validates model loading and tensor inference only; it does not measure CCTV
identity recovery accuracy.
"""
import sys
from pathlib import Path

# When invoked as `python tools/verify_osnet.py`, Python places the tools/
# directory (not the project root) on sys.path. Add the project root so the
# sibling `argus` package can be imported reliably on Windows.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from argus.features import OSNetExtractor, appearance_similarity


def main():
    extractor = OSNetExtractor(device="cpu", model_name="osnet_x0_25")
    frame = np.zeros((360, 240, 3), dtype=np.uint8)
    feature = extractor.extract(frame, (40, 20, 200, 340))
    norm = float(np.linalg.norm(feature))
    same_score = appearance_similarity(feature, feature)
    print("OSNET CHECKPOINT:", extractor.checkpoint_path)
    print("EMBEDDING DIMENSION:", feature.size)
    print("EMBEDDING L2 NORM:", round(norm, 6))
    print("SELF COSINE SIMILARITY:", round(same_score, 6))
    if not np.all(np.isfinite(feature)) or not np.isclose(norm, 1.0, atol=1e-3):
        raise SystemExit("FAIL: invalid OSNet embedding")
    if not np.isclose(same_score, 1.0, atol=1e-3):
        raise SystemExit("FAIL: cosine similarity sanity check failed")
    print("OSNET INFERENCE SMOKE TEST: PASS")
    print("NOTE: This is not a real-person ReID accuracy test.")


if __name__ == "__main__":
    main()
