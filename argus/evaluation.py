"""Evaluation helpers. Full benchmark metrics require ground-truth MOT annotations."""
import pandas as pd
from pathlib import Path

def load_motchallenge(path):
    """Read MOTChallenge GT/detection text: frame,id,x,y,w,h,conf,...
    Returns a normalized DataFrame. Coordinates are converted to xyxy.
    """
    df = pd.read_csv(path, header=None)
    if df.shape[1] < 6:
        raise ValueError("Expected MOTChallenge columns: frame,id,x,y,w,h,...")
    df = df.iloc[:, :7].copy()
    while df.shape[1] < 7:
        df[df.shape[1]] = 1.0
    df.columns = ["frame","identity_id","x","y","w","h","confidence"]
    df["x2"] = df["x"] + df["w"]
    df["y2"] = df["y"] + df["h"]
    return df

def evaluate_with_motmetrics(gt_path, prediction_path):
    """Compute standard MOT metrics when motmetrics is installed."""
    try:
        import motmetrics as mm
    except ImportError as exc:
        raise RuntimeError("Install motmetrics: pip install motmetrics") from exc
    gt = load_motchallenge(gt_path)
    pred = pd.read_csv(prediction_path)
    acc = mm.MOTAccumulator(auto_id=True)
    for frame in sorted(set(gt.frame) | set(pred.frame)):
        g = gt[gt.frame == frame]
        p = pred[pred.frame == frame]
        g_ids = g.identity_id.astype(int).tolist()
        p_ids = p.identity_id.astype(int).tolist()
        distances = mm.distances.iou_matrix(
            g[["x","y","x2","y2"]].to_numpy(),
            p[["x1","y1","x2","y2"]].to_numpy(),
            max_iou=0.5
        )
        acc.update(g_ids, p_ids, distances)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=[
        "idf1","idp","idr","mota","motp","num_switches","num_fragmentations"
    ], name="ARGUS-X")
    return summary
