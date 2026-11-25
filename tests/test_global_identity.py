import numpy as np
from argus.global_identity import GlobalIdentityManager

def test_same_local_track_keeps_global_id():
    m=GlobalIdentityManager(threshold=.5,margin=.05); f=np.ones(8,dtype=np.float32)
    a=m.resolve("CAM_01",1,f,0.0)[0]; b=m.resolve("CAM_01",1,f,1.0)[0]
    assert a==b

def test_cross_camera_appearance_match_can_recover_global_id():
    m=GlobalIdentityManager(threshold=.5,margin=.05); f=np.array([1,0,0],dtype=np.float32)
    a=m.resolve("CAM_01",1,f,0.0)[0]; b,status,_=m.resolve("CAM_02",3,f,2.0)
    assert a==b and status=="cross_camera_recovered"

def test_unmatched_person_gets_new_global_id():
    m=GlobalIdentityManager(threshold=.95,margin=.1)
    a=m.resolve("CAM_01",1,np.array([1,0],dtype=np.float32),0)[0]
    b=m.resolve("CAM_02",1,np.array([0,1],dtype=np.float32),0)[0]
    assert a!=b


def test_same_camera_new_local_track_recovers_existing_global_id():
    """A reappearing person with a new local track can retain its global ID."""
    m = GlobalIdentityManager(threshold=0.72, margin=0.08, max_gap_seconds=120.0)
    feature = np.zeros(512, dtype=np.float32)
    feature[17] = 1.0

    original_id, _, _ = m.resolve("CAM_01", 1, feature, 0.0)
    recovered_id, status, score = m.resolve("CAM_01", 2, feature.copy(), 3.0)

    assert recovered_id == original_id
    assert status == "cross_camera_recovered"
    assert score >= 0.72
