import numpy as np

from manga_localizer.lettering_audit import inspect_page


def test_residual_inside_authorized_region_and_near_the_border():
    base = np.zeros((95, 120, 3), np.uint8)
    face = [10, 10, 110, 85]
    base[10:85, 10:110] = 255
    # The near-white bridge opens the second glyph into the edge at a strict threshold.
    base[59, 10:23] = 242
    base[58:62, 23:28] = 48
    base[30:50, 45:48] = 35
    donor = np.full_like(base, 255)
    donor[30:50, 45:48] = 35
    donor[28:47, 60:65] = 10
    master = base.copy()
    master[10:85, 10:110] = donor[10:85, 10:110]
    faces = [dict(bubble=face, text=[19, 22, 69, 67])]
    groups = [dict(label="reviewed", status="applied", erase=[19, 22, 69, 67], ink=[37, 21, 75, 58])]
    def inspect():
        return inspect_page(base, master, donor, np.eye(2, 3), groups, faces)

    assert inspect() == dict(checked_faces=1, unmatched=[], findings=[])
    master[58:62, 23:28] = base[58:62, 23:28]
    finding, = inspect()["findings"]
    assert (finding["box"], finding["pixels"], finding["groups"]) == ([23, 58, 28, 62], 20, ["reviewed"])
    master[58, 23, 0] = 49
    assert inspect()["findings"][0]["signature"] != finding["signature"]
    assert inspect_page(base, master, donor, np.eye(2, 3), groups, [])["unmatched"] == ["reviewed"]
