"""Single-camera tracker with appearance memory and cautious long-term recovery."""
from dataclasses import dataclass, field
from collections import deque
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from .features import extract_hsv_appearance, appearance_similarity
from .motion import iou, center, bbox_diagonal, transform_bbox

@dataclass
class Track:
    identity_id: int
    class_id: int
    class_name: str
    bbox: np.ndarray
    appearance: np.ndarray
    confidence: float
    first_frame: int
    last_frame: int
    hits: int = 1
    state: str = "tentative"
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    history: deque = field(default_factory=lambda: deque(maxlen=500))
    feature_bank: deque = field(default_factory=lambda: deque(maxlen=20))
    recovery_count: int = 0

    def __post_init__(self):
        self.bbox = np.asarray(self.bbox, dtype=np.float32)
        self.history.append((self.last_frame, *center(self.bbox).tolist()))
        if self.appearance is not None and np.any(self.appearance):
            self.feature_bank.append(self.appearance.copy())

    def predict(self, frame_index, camera_matrix=None):
        dt = max(1, frame_index - self.last_frame)
        predicted = transform_bbox(self.bbox, camera_matrix)
        shift = self.velocity * min(dt, 20)
        predicted[[0, 2]] += shift[0]
        predicted[[1, 3]] += shift[1]
        return predicted

    def update(self, detection, frame_index):
        old_center = center(self.bbox)
        new_bbox = np.asarray(detection["bbox"], dtype=np.float32)
        new_center = center(new_bbox)
        dt = max(1, frame_index - self.last_frame)
        measured = (new_center - old_center) / dt
        self.velocity = 0.65 * self.velocity + 0.35 * measured
        self.bbox = new_bbox
        feature = detection.get("appearance")
        if feature is not None and np.any(feature):
            if self.appearance is None or not np.any(self.appearance):
                self.appearance = feature.copy()
            else:
                self.appearance = 0.90 * self.appearance + 0.10 * feature
            self.feature_bank.append(feature.copy())
        self.confidence = float(detection["confidence"])
        self.last_frame = frame_index
        self.hits += 1
        self.state = "active" if self.hits >= 2 else "tentative"
        self.history.append((frame_index, *new_center.tolist()))

    def best_appearance_similarity(self, feature):
        if feature is None or not np.any(feature):
            return 0.0
        scores = [appearance_similarity(feature, old) for old in self.feature_bank]
        if self.appearance is not None:
            scores.append(appearance_similarity(feature, self.appearance))
        return max(scores, default=0.0)

class IdentityTracker:
    """Per-camera local identity tracker. Global IDs are assigned separately."""
    def __init__(self, max_active_age=30, max_lost_age=600,
                 recovery_threshold=0.70, recovery_margin=0.10,
                 min_hits=2, iou_threshold=0.30, feature_extractor=None):
        self.max_active_age=max_active_age; self.max_lost_age=max_lost_age
        self.recovery_threshold=recovery_threshold; self.recovery_margin=recovery_margin
        self.min_hits=min_hits; self.iou_threshold=iou_threshold
        self.feature_extractor=feature_extractor
        self.tracks={}; self.next_id=itertools.count(1); self.events=[]; self.frame_index=0

    def _prepare_detections(self, frame, detections):
        prepared=[]
        for det in detections:
            if det.get("class_name") != "person" and int(det.get("class_id", 0)) != 0:
                continue
            item=dict(det); item["class_id"]=0; item["class_name"]="person"
            item["bbox"]=np.asarray(item["bbox"], dtype=np.float32)
            item["appearance"]=(self.feature_extractor.extract(frame, item["bbox"]) if self.feature_extractor is not None else extract_hsv_appearance(frame, item["bbox"]))
            prepared.append(item)
        return prepared

    def _active_cost(self, track, det, predicted):
        overlap=iou(predicted, det["bbox"])
        dist=np.linalg.norm(center(predicted)-center(det["bbox"]))
        diag=max(bbox_diagonal(predicted),bbox_diagonal(det["bbox"]),30.0)
        nd=min(1.0,dist/(diag*2.5))
        app=track.best_appearance_similarity(det["appearance"])
        if overlap < 0.01 and nd > 0.85:
            return 1e6
        return 0.45*(1-overlap)+0.30*nd+0.25*(1-app)

    def update(self, frame, detections, frame_index=None, camera_matrix=None):
        if frame_index is None: self.frame_index += 1; frame_index=self.frame_index
        else: self.frame_index=frame_index
        dets=self._prepare_detections(frame,detections)
        for old_track in self.tracks.values():
            if old_track.state in ("active", "tentative") and frame_index-old_track.last_frame > self.max_active_age:
                old_track.state="lost"
                self.events.append({"frame":frame_index,"identity_id":old_track.identity_id,"event":"lost","confidence":old_track.confidence,"detail":"Exceeded active-track age limit."})
        active=[t for t in self.tracks.values() if t.state in ("active","tentative") and frame_index-t.last_frame<=self.max_active_age]
        matched_t=set(); matched_d=set()
        predictions={t.identity_id:t.predict(frame_index,camera_matrix) for t in active}
        if active and dets:
            cost=np.full((len(active),len(dets)),1e6,dtype=np.float32)
            for r,t in enumerate(active):
                for c,d in enumerate(dets): cost[r,c]=self._active_cost(t,d,predictions[t.identity_id])
            rows,cols=linear_sum_assignment(cost)
            for r,c in zip(rows,cols):
                if cost[r,c] < 0.92:
                    t=active[r]; t.update(dets[c],frame_index)
                    matched_t.add(t.identity_id); matched_d.add(c)
        for t in active:
            if t.identity_id not in matched_t and t.state not in ("lost","expired"):
                # Keep a short missed-detection grace window before entering long-term lost state.
                if frame_index - t.last_frame >= min(5, self.max_active_age):
                    t.state="lost"
                    self.events.append({"frame":frame_index,"identity_id":t.identity_id,"event":"lost","confidence":t.confidence,"detail":"Track missed the short grace window."})
        lost=[t for t in self.tracks.values() if t.state=="lost" and 1<=frame_index-t.last_frame<=self.max_lost_age]
        remaining=[i for i in range(len(dets)) if i not in matched_d]
        if lost and remaining:
            scores=np.zeros((len(lost),len(remaining)),dtype=np.float32)
            for r,t in enumerate(lost):
                pred=t.predict(frame_index,camera_matrix); gap=frame_index-t.last_frame
                for cc,di in enumerate(remaining):
                    d=dets[di]; app=t.best_appearance_similarity(d["appearance"])
                    dist=np.linalg.norm(center(pred)-center(d["bbox"]))
                    scale=max(bbox_diagonal(pred),bbox_diagonal(d["bbox"]),50.0)
                    motion=float(np.exp(-dist/(scale*(2.0+gap/25.0))))
                    temporal=float(np.exp(-gap/max(1.0,self.max_lost_age)))
                    scores[r,cc]=0.72*app+0.20*motion+0.08*temporal
            rows,cols=linear_sum_assignment(1.0-scores)
            for r,cc in zip(rows,cols):
                score=float(scores[r,cc]); col_scores=np.sort(scores[:,cc])[::-1]
                second=float(col_scores[1]) if len(col_scores)>1 else 0.0
                if score < self.recovery_threshold or score-second < self.recovery_margin: continue
                t=lost[r]; di=remaining[cc]; gap=frame_index-t.last_frame
                t.update(dets[di],frame_index); t.recovery_count+=1
                matched_t.add(t.identity_id); matched_d.add(di)
                self.events.append({"frame":frame_index,"identity_id":t.identity_id,"event":"recovered","confidence":score,"detail":f"Recovered after {gap} frame(s)."})
        for i,d in enumerate(dets):
            if i in matched_d: continue
            identity=next(self.next_id)
            t=Track(identity,0,"person",d["bbox"],d["appearance"],d["confidence"],frame_index,frame_index)
            self.tracks[identity]=t
            self.events.append({"frame":frame_index,"identity_id":identity,"event":"created","confidence":d["confidence"],"detail":"New person identity created."})
        for t in self.tracks.values():
            if t.state=="lost" and frame_index-t.last_frame>self.max_lost_age: t.state="expired"
        # Include tentative tracks: the previous implementation accidentally hid them.
        visible=[t for t in self.tracks.values() if t.last_frame==frame_index and t.state in ("active","tentative")]
        return visible

    def summary(self):
        return {"unique_identities":len(self.tracks),"active":sum(t.state in ("active","tentative") for t in self.tracks.values()),"lost":sum(t.state=="lost" for t in self.tracks.values()),"recovered":sum(t.recovery_count>0 for t in self.tracks.values()),"recovery_events":sum(e["event"]=="recovered" for e in self.events)}
