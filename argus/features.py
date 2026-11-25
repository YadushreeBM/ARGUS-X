"""Appearance features for ARGUS-X.

Phase A uses Torchreid OSNet x0.25 trained for person re-identification
on Market-1501. ImageNet initialization is deliberately NOT used as a
substitute for a person-ReID checkpoint.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import cv2
import numpy as np


# Official Torchreid Model Zoo: osnet_x0_25 trained on Market-1501.
# See https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO
OSNET_MARKET1501_GDRIVE_ID = "1z1UghYvOTtjx7kEoRfmqSMu-z62J6MAj"
OSNET_MARKET1501_FILENAME = "osnet_x0_25_market1501.pth.tar"
MARKET1501_NUM_TRAIN_IDENTITIES = 751


def _region_hist(hsv, bins=(16, 8, 8)):
    parts = []
    for channel, count, limits in (
        (0, bins[0], [0, 180]),
        (1, bins[1], [0, 256]),
        (2, bins[2], [0, 256]),
    ):
        hist = cv2.calcHist([hsv], [channel], None, [count], limits).flatten()
        hist = hist / (float(hist.sum()) + 1e-8)
        parts.append(hist.astype(np.float32))
    return np.concatenate(parts)


def extract_hsv_appearance(frame, bbox, bins=(16, 8, 8)):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros(sum(bins) * 3, dtype=np.float32)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(sum(bins) * 3, dtype=np.float32)
    crop = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    whole = _region_hist(hsv, bins)
    upper = _region_hist(hsv[:64], bins)
    lower = _region_hist(hsv[64:], bins)
    return np.concatenate((whole, upper, lower)).astype(np.float32)


class OSNetExtractor:
    """CPU OSNet x0.25 feature extractor with Market-1501 ReID weights."""

    def __init__(
        self,
        device="cpu",
        model_name="osnet_x0_25",
        checkpoint_path=None,
        download_if_missing=True,
    ):
        try:
            import torch
            from torchreid.reid.models import build_model
        except Exception as exc:
            raise RuntimeError(
                "OSNet dependencies are unavailable. Run "
                "'python -m pip install -r requirements.txt' and verify "
                "that torch, torchvision and torchreid import successfully. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

        self.torch = torch
        self.device = torch.device(device)
        project_root = Path(__file__).resolve().parents[1]
        weights_dir = project_root / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else (
            weights_dir / OSNET_MARKET1501_FILENAME
        )

        if not self.checkpoint_path.is_file():
            if not download_if_missing:
                raise FileNotFoundError(
                    f"Person-ReID checkpoint not found: {self.checkpoint_path}"
                )
            self._download_checkpoint()

        # pretrained=False is intentional: build_model(pretrained=True) loads
        # ImageNet weights, not the Market-1501 person-ReID checkpoint.
        self.model = build_model(
            name=model_name,
            num_classes=MARKET1501_NUM_TRAIN_IDENTITIES,
            loss="softmax",
            pretrained=False,
        )
        self._load_reid_checkpoint()
        self.model.to(self.device).eval()

        self.mean = torch.tensor(
            [0.485, 0.456, 0.406], device=self.device
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.229, 0.224, 0.225], device=self.device
        ).view(1, 3, 1, 1)
        self._inference_lock = threading.Lock()
        self.embedding_dim = None

    def _download_checkpoint(self):
        try:
            import gdown
        except Exception as exc:
            raise RuntimeError(
                "The gdown package is required to download the official "
                "Torchreid Market-1501 checkpoint. Install gdown."
            ) from exc

        print(
            "ARGUS-X: downloading OSNet x0.25 Market-1501 ReID checkpoint "
            "to", self.checkpoint_path
        )
        downloaded = gdown.download(
            id=OSNET_MARKET1501_GDRIVE_ID,
            output=str(self.checkpoint_path),
            quiet=False,
        )
        if not downloaded or not self.checkpoint_path.is_file():
            raise RuntimeError(
                "Could not download the official OSNet Market-1501 checkpoint. "
                "Check internet access or manually place the checkpoint at "
                f"{self.checkpoint_path}"
            )

    def _load_reid_checkpoint(self):
        # The checkpoint is from the official Torchreid model zoo. Support
        # common Torchreid checkpoint container formats and DataParallel keys.
        checkpoint = self.torch.load(
            str(self.checkpoint_path),
            map_location="cpu",
            weights_only=False,
        )
        if isinstance(checkpoint, dict):
            state = (
                checkpoint.get("state_dict")
                or checkpoint.get("model_state_dict")
                or checkpoint.get("model")
                or checkpoint
            )
        else:
            raise RuntimeError("Unexpected OSNet checkpoint format.")

        if not isinstance(state, dict):
            raise RuntimeError("OSNet checkpoint does not contain a state dict.")

        cleaned = {}
        for key, value in state.items():
            name = str(key)
            for prefix in ("module.", "model."):
                if name.startswith(prefix):
                    name = name[len(prefix):]
            cleaned[name] = value

        incompatible = self.model.load_state_dict(cleaned, strict=False)
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)

        # A valid 751-class Market-1501 checkpoint should load the backbone
        # and classifier. Do not silently accept a mostly-unloaded model.
        allowed_missing = []
        allowed_unexpected = []
        bad_missing = [k for k in missing if k not in allowed_missing]
        bad_unexpected = [k for k in unexpected if k not in allowed_unexpected]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                "OSNet ReID checkpoint/model mismatch. "
                f"Missing keys ({len(bad_missing)}): {bad_missing[:8]}; "
                f"unexpected keys ({len(bad_unexpected)}): {bad_unexpected[:8]}"
            )

    def extract(self, frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros(self.embedding_dim or 512, dtype=np.float32)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(self.embedding_dim or 512, dtype=np.float32)

        # Torchreid OSNet model-zoo inference resolution: H=256, W=128.
        rgb = cv2.cvtColor(
            cv2.resize(crop, (128, 256), interpolation=cv2.INTER_LINEAR),
            cv2.COLOR_BGR2RGB,
        )
        arr = np.ascontiguousarray(
            rgb.transpose(2, 0, 1), dtype=np.float32
        ) / 255.0
        tensor = self.torch.from_numpy(arr).unsqueeze(0).to(self.device)
        tensor = (tensor - self.mean) / self.std

        with self._inference_lock, self.torch.inference_mode():
            embedding = self.model(tensor)
            if isinstance(embedding, (tuple, list)):
                embedding = embedding[0]
            if embedding.ndim != 2 or embedding.shape[0] != 1:
                raise RuntimeError(
                    f"Unexpected OSNet output shape: {tuple(embedding.shape)}"
                )
            embedding = self.torch.nn.functional.normalize(embedding, dim=1)
            result = embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)

        if not np.all(np.isfinite(result)):
            raise RuntimeError("OSNet produced a non-finite embedding.")
        self.embedding_dim = int(result.size)
        return result


def appearance_similarity(a, b):
    """Cosine similarity for deep embeddings; Bhattacharyya for HSV histograms."""
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size != b.size or not a.size or not np.any(a) or not np.any(b):
        return 0.0
    if a.size >= 256:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.clip(np.dot(a, b) / denom if denom else 0.0, 0.0, 1.0))
    distance = cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)
    return float(np.clip(1.0 - distance, 0.0, 1.0))
