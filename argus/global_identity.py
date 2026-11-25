"""Conservative cross-camera global identity registry.

This is an appearance-led association layer, not biometric identification. It only
matches person tracks from supplied camera streams when score and ambiguity gates pass.
"""
from dataclasses import dataclass, field
from collections import deque
from .features import appearance_similarity

@dataclass
class GlobalIdentity:
    global_id: str
    feature_bank: deque = field(default_factory=lambda: deque(maxlen=30))
    last_seen_time: float = 0.0
    last_camera: str = ""
    active_cameras: set = field(default_factory=set)
    observations: int = 0

    def similarity(self, feature):
        return max((appearance_similarity(feature, f) for f in self.feature_bank), default=0.0)

class GlobalIdentityManager:
    def __init__(self, threshold=0.78, margin=0.10, max_gap_seconds=30.0):
        self.threshold=threshold; self.margin=margin; self.max_gap_seconds=max_gap_seconds
        self.identities={}; self.local_to_global={}; self.events=[]; self._next=1

    def resolve(self, camera_id, local_track_id, feature, timestamp, track_state="active"):
        key=(str(camera_id),int(local_track_id))
        if key in self.local_to_global:
            gid=self.local_to_global[key]
            identity=self.identities[gid]
            if feature is not None and getattr(feature,"size",0): identity.feature_bank.append(feature.copy())
            identity.last_seen_time=max(identity.last_seen_time,float(timestamp))
            identity.last_camera=str(camera_id); identity.active_cameras.add(str(camera_id)); identity.observations+=1
            return gid,"continued",1.0
        candidates=[]
        for gid,identity in self.identities.items():
            gap=abs(float(timestamp)-identity.last_seen_time)
            # Do not merge two simultaneous local tracks from the same camera.
            if str(camera_id) in identity.active_cameras and gap < 1.0: continue
            if gap > self.max_gap_seconds: continue
            score=identity.similarity(feature) if feature is not None else 0.0
            candidates.append((score,gid))
        candidates.sort(reverse=True)
        if candidates:
            best=candidates[0][0]; second=candidates[1][0] if len(candidates)>1 else 0.0
            if best>=self.threshold and best-second>=self.margin:
                gid=candidates[0][1]; ident=self.identities[gid]
                self.local_to_global[key]=gid
                if feature is not None: ident.feature_bank.append(feature.copy())
                ident.last_seen_time=max(ident.last_seen_time,float(timestamp)); ident.last_camera=str(camera_id)
                ident.active_cameras.add(str(camera_id)); ident.observations+=1
                self.events.append({"timestamp":float(timestamp),"camera_id":str(camera_id),"local_track_id":int(local_track_id),"global_id":gid,"event":"cross_camera_recovered","confidence":float(best)})
                return gid,"cross_camera_recovered",float(best)
        gid=f"PERSON_{self._next:04d}"; self._next+=1
        ident=GlobalIdentity(gid,last_seen_time=float(timestamp),last_camera=str(camera_id),active_cameras={str(camera_id)},observations=1)
        if feature is not None: ident.feature_bank.append(feature.copy())
        self.identities[gid]=ident; self.local_to_global[key]=gid
        self.events.append({"timestamp":float(timestamp),"camera_id":str(camera_id),"local_track_id":int(local_track_id),"global_id":gid,"event":"global_identity_created","confidence":1.0})
        return gid,"created",1.0

    def close_camera_frame(self, camera_id):
        # The caller may use this when a camera has no current observation. We keep
        # mappings stable; active_cameras is a conservative recent-presence hint.
        pass
