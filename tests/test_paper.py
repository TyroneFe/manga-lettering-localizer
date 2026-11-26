from hashlib import sha256

import numpy as np
import pytest
from PIL import Image

from manga_localizer.lettering import composite, propose, reviewed_paper
from manga_localizer.paper import include_boundary_glyphs, read_stroke_mask, sampled_proposal


def test_transfer_restores_sampled_background_and_preserves_nearby_art():
    base = np.full((50, 100, 3), 255, dtype=np.uint8)
    base[17:] = [251, 252, 253]
    base[20:30, 33:40] = 30
    base[20:30, 60:65] = [90, 30, 20]
    donor = np.full_like(base, 255)
    donor[20:30, 43:50] = 5
    erase = [26, 12, 42, 37]
    ink = [41, 12, 55, 37]
    paper = reviewed_paper(base, erase, ink, [[3, 16], [80, 96]])
    clean, gray = propose(base, donor, np.eye(2, 3), erase, ink, paper=paper)
    result, _ = composite(base, [(erase, ink, clean, gray, paper)])
    assert np.all(result[20:30, 33:40] == [251, 252, 253])
    assert np.all(result[20:30, 43:50] == 5)
    assert np.array_equal(result[:, 55:], base[:, 55:])

    with pytest.raises(ValueError, match="do not agree"):
        reviewed_paper(base, erase, ink, [[3, 16], [60, 65]])


def test_local_paper_matches_known_gradient_and_registered_ink():
    # A slight paper gradient under separated strokes, with a clean reference image.
    y, x = np.indices((100, 130))
    value = 249 + (x // 30) + (y // 55)
    reference = np.stack((value, value+1, value), axis=2).astype(np.uint8)
    base = reference.copy()
    base[31:59, 32:37] = 32
    base[53:59, 33:55] = 44
    base[65:69, 53:57] = 10
    donor = np.full_like(base, 255)
    donor[33:66, 78:81] = 128
    donor[33:66, 81:85] = 0
    donor[72:76, 81:85] = 0
    region = dict(erase=[9, 14, 72, 84], ink=[67, 18, 103, 86])
    proposal, metrics = sampled_proposal(base, donor, np.eye(2, 3), region)
    actual, _ = composite(base, [proposal])
    expected = ((reference.astype(np.uint16) * donor + 127) // 255).astype(np.uint8)
    strokes = np.any(base != reference, axis=2)
    np.testing.assert_array_equal(actual[~strokes], expected[~strokes])
    assert np.abs(actual[strokes].astype(int) - expected[strokes]).max() <= 1
    assert metrics["samples"] > 1000 and metrics["p99"] <= 1

    # Art beneath the proposed donor is independently checked even when the erase field fits.
    base[42:45, 77:80] = [90, 30, 20]
    with pytest.raises(ValueError, match="beneath"):
        sampled_proposal(base, donor, np.eye(2, 3), region)


def test_local_paper_holds_patterned_paper_and_chromatic_details():
    y, x = np.indices((120, 120))
    donor = np.full((120, 120, 3), 255, np.uint8)
    donor[45:65, 76:80] = 0
    base = donor.copy()
    # The regular dot pattern is source artwork even though all its tones are light.
    base[:] = 255
    base[((x // 8 + y // 8) % 2 == 0) & (x > 24) & (x < 68) & (y > 25) & (y < 84)] = 245
    base[30:48, 34:39] = 40
    region = dict(erase=[12, 12, 85, 97], ink=[67, 18, 100, 91])
    with pytest.raises(ValueError, match="block checks"):
        sampled_proposal(base, donor, np.eye(2, 3), region)
    base[33, 37] = [95, 30, 20]
    with pytest.raises(ValueError, match="chromatic"):
        sampled_proposal(base, donor, np.eye(2, 3), region)


def test_reviewed_boundary_stroke_preserves_nearby_line_and_paper():
    base = np.full((90, 120, 3), 254, np.uint8)
    base[56, 48:68] = 35
    base[41, 57] = 255
    donor = np.full_like(base, 255)
    donor[56, 48:68] = 35
    reference = base.copy()
    base[42:55, 58:62] = 40
    base[42:55, [57, 62]] = 246
    erase, ink = [35, 24, 82, 70], [85, 35, 99, 70]
    shape = [[57, 41], [62, 41], [62, 54], [57, 54]]
    detail = dict(seed=[59, 46], shape=shape)
    region = dict(boundary_glyphs=[detail])
    before = (erase, ink, np.zeros((46, 47), bool), np.full((35, 14), 255, np.uint8))
    proposal, support = include_boundary_glyphs(base, donor, np.eye(2, 3), region, before)
    actual, _ = composite(base, [proposal])
    np.testing.assert_array_equal(actual, reference)
    assert support.sum() == 6 * 14

    with pytest.raises(ValueError, match="overlaps"):
        include_boundary_glyphs(base, donor, np.eye(2, 3),
                                dict(boundary_glyphs=[detail], paper_marks=[[57, 41, 59, 43]]), before)
    donor[44, 59] = 40
    with pytest.raises(ValueError, match="donor"):
        include_boundary_glyphs(base, donor, np.eye(2, 3), region, before)
    donor[44, 59] = 255
    detail["shape"] = [[57, 41], [62, 41], [62, 56], [57, 56]]
    with pytest.raises(ValueError, match="protected artwork"):
        include_boundary_glyphs(base, donor, np.eye(2, 3), region, before)


def test_segmented_strokes_leave_artwork_between_characters_untouched():
    reference = np.full((120, 150, 3), 254, np.uint8)
    reference[:, 67:72] = [45, 70, 35]
    reference[78:81, 88:90] = 235
    base = reference.copy()
    base[34:52, 37:42] = 36
    base[58:65, 91:96] = 32
    donor = np.full_like(base, 255)
    donor[76:87, 87:90] = 128
    donor[76:87, 90:93] = 0
    region = dict(erase=[14, 10, 125, 105], ink=[77, 69, 111, 96],
                  paper_marks=[[86, 76, 92, 83]])
    full = np.zeros(base.shape[:2], bool)
    full[31:55, 34:45] = True
    full[55:68, 88:99] = True
    mask = full[10:105, 14:125]
    proposal, metrics = sampled_proposal(base, donor, np.eye(2, 3), region, mask)
    result, _ = composite(base, [proposal])
    expected = ((reference.astype(np.uint16) * donor + 127) // 255).astype(np.uint8)
    np.testing.assert_array_equal(result, expected)
    assert metrics["worst"] == 0
    full[34, 37] = False
    with pytest.raises(ValueError, match="clips original ink"):
        sampled_proposal(base, donor, np.eye(2, 3), region, mask)
    full[34, 37] = True
    full[48, 66] = True
    with pytest.raises(ValueError, match="unrelated artwork"):
        sampled_proposal(base, donor, np.eye(2, 3), region, mask)


def test_stroke_mask_is_bound_to_reviewed_bytes_and_window(tmp_path):
    values = np.zeros((18, 20), np.uint8)
    values[5:9, 7:11] = 255
    path = tmp_path / "mask.png"
    Image.fromarray(values).save(path)
    spec = dict(file=path.name, sha256=sha256(path.read_bytes()).hexdigest())
    region = dict(erase=[30, 40, 50, 58], stroke_mask=spec)
    np.testing.assert_array_equal(read_stroke_mask(tmp_path, region), values != 0)
    with pytest.raises(ValueError, match="representation"):
        read_stroke_mask(tmp_path, dict(region, erase=[30, 40, 51, 58]))
    values[0, 0] = 64
    Image.fromarray(values).save(path)
    with pytest.raises(ValueError, match="hash"):
        read_stroke_mask(tmp_path, region)
    spec["sha256"] = sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="samples"):
        read_stroke_mask(tmp_path, region)
