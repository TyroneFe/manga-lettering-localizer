import cv2
import numpy as np
import pytest

from manga_localizer.lettering import composite, lettering_placement, propose
from manga_localizer.render import comparison_window
from manga_localizer.registration import correspondence

IDENTITY=np.array([[1.,0,0],[0,1.,0]])


def test_flat_ink_and_outside_identity():
    base=np.full((120,160,3),255,np.uint8)
    base[:20]=np.arange(160,dtype=np.uint8)[None,:,None]
    base[49:67,50:54]=0
    donor=np.full_like(base,255)
    cv2.putText(donor,"A!",(61,66),cv2.FONT_HERSHEY_SIMPLEX,.75,(0,0,0),2,cv2.LINE_AA)
    erase=[42,41,60,74]
    ink=[54,39,100,75]
    clean,gray=propose(base,donor,IDENTITY,erase,ink)
    out,allowed=composite(base,[(erase,ink,clean,gray)])
    assert np.array_equal(out[~allowed],base[~allowed])
    assert out[50:65,51].min()==255
    assert out[42:72,64:89].min()<10
    assert np.any((gray>0)&(gray<255))
    assert base[55,52].max()==0


def test_local_scale_preserves_proportions_and_diagnostic_includes_original_donor():
    registration = np.array([[0.6, -0.02, 12], [0.02, 0.6, 5]])
    region = dict(placement_scale=0.8, placement_anchor=[80, 105], placement_offset=[5, -20],
                  erase=[64, 70, 102, 108], ink=[50, 60, 126, 123], status="applied")
    placement = lettering_placement(registration, region)
    points = np.array([[0, 0], [80, 20], [150, 305]])
    registered = points @ registration[:, :2].T + registration[:, 2]
    expected = (registered - [80, 105]) * 0.8 + [85, 85]
    np.testing.assert_allclose(points @ placement[:, :2].T + placement[:, 2], expected)
    np.testing.assert_array_equal(lettering_placement(registration, {}), registration)
    ys, xs = comparison_window(region, (300, 200, 3))
    assert ys.stop >= 153 and xs.stop >= 132
    with pytest.raises(ValueError, match="placement"):
        lettering_placement(registration, {"placement_scale": 0})


def test_reviewed_paper_marks_preserve_base_under_donor_ink():
    base = np.full((70, 90, 3), 255, np.uint8)
    donor = base.copy()
    base[25:41, 22:25] = 0
    base[32:34, 39:43] = 234
    donor[25:41, 41:45] = 128
    donor[25:41, 45:47] = 0
    erase, ink = [12, 15, 54, 50], [30, 18, 54, 50]
    mark = [37, 30, 44, 36]
    with pytest.raises(ValueError, match="nonuniform"):
        propose(base, donor, IDENTITY, erase, ink)
    clean, gray = propose(base, donor, IDENTITY, erase, ink, paper_marks=[mark])
    out, _ = composite(base, [(erase, ink, clean, gray)])
    assert np.all(out[32, 39] == 234)
    assert np.all(out[32, 41] == (234 * 128 + 127) // 255)
    assert np.all(out[32, 23] == 255)
    base[32, 39] = [234, 255, 234]
    with pytest.raises(ValueError, match="paper mark"):
        propose(base, donor, IDENTITY, erase, ink, paper_marks=[mark])


@pytest.mark.parametrize("bad",["perimeter","art","clipped"])
def test_region_failure_is_nondestructive(bad):
    base=np.full((85,100,3),255,np.uint8)
    donor=base.copy()
    base[30:52,32:36]=0
    donor[30:52,60:65]=0
    if bad=="perimeter": base[22,29]=0
    if bad=="art": base[33,61]=(255,20,0)
    if bad=="clipped": donor[22:40,60]=0
    before=base.copy()
    with pytest.raises(ValueError):
        propose(base,donor,IDENTITY,[23,22,45,61],[49,22,77,61])
    assert np.array_equal(before,base)


def test_affine_direction_and_holdout():
    rng=np.random.default_rng(912)
    points=rng.uniform(0,1100,(200,2)).astype(np.float32)
    desc=rng.random((200,128)).astype(np.float32)
    expected=np.array([[.55,-.02,123],[.01,.54,26]])
    dest=(points@expected[:,:2].T+expected[:,2]).astype(np.float32)
    dest[5::17]+=400
    result=correspondence((points,desc),(dest,desc.copy()))
    np.testing.assert_allclose(result["matrix"],expected,atol=.0001)
    assert len(result["held_errors"])>=30 and result["held_errors"].max()<.01
    assert correspondence((points,None),(dest,desc)) is None
