import numpy as np
from argus.tracker import IdentityTracker

def frame(): return np.zeros((240,320,3),dtype=np.uint8)
def det(x1=20,x2=80): return {"bbox":[x1,20,x2,120],"confidence":.9,"class_id":0,"class_name":"person"}

def test_new_person_is_visible_immediately():
    t=IdentityTracker(min_hits=2); visible=t.update(frame(),[det()],1)
    assert len(visible)==1 and visible[0].identity_id==1 and visible[0].state=="tentative"

def test_local_track_continues():
    t=IdentityTracker(); t.update(frame(),[det()],1)
    visible=t.update(frame(),[det(22,82)],2)
    assert len(visible)==1 and visible[0].identity_id==1 and visible[0].hits==2

def test_non_person_detections_are_filtered():
    t=IdentityTracker(); car={"bbox":[100,20,180,120],"confidence":.9,"class_id":2,"class_name":"car"}
    assert t.update(frame(),[car],1)==[] and len(t.tracks)==0

def test_short_missed_detection_uses_grace_window_then_becomes_lost():
    t=IdentityTracker(max_active_age=2); t.update(frame(),[det()],1)
    t.update(frame(),[],2)
    assert t.tracks[1].state == "tentative"
    for idx in range(3, 8):
        t.update(frame(), [], idx)
    assert t.tracks[1].state == "lost"
