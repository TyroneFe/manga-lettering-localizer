import cv2
import numpy as np
import pytest

from manga_localizer import discover as detector
from manga_localizer.discover import interiors
from manga_localizer.lettering import composite, propose


def test_text_windows_ignore_blank_interiors_and_page_border():
    page = np.zeros((360, 400, 3), np.uint8)
    cv2.rectangle(page, (20, 35), (140, 135), (255, 255, 255), -1)
    cv2.putText(page, "HI!", (32, 104), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 4)
    cv2.rectangle(page, (180, 35), (300, 135), (255, 255, 255), -1)
    cv2.rectangle(page, (0, 200), (130, 330), (255, 255, 255), -1)
    cv2.putText(page, "HI!", (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 4)

    (candidate,) = interiors(page)
    assert candidate["bubble"] == [20, 35, 141, 136]
    x, y, right, bottom = candidate["text"]
    assert 20 < x < 40 and 35 < y < 83 and 83 < right < 141 and 104 < bottom < 136


def test_fitted_proposal_preserves_bubble_border_and_disconnected_punctuation(monkeypatch):
    base = np.zeros((320, 400, 3), np.uint8)
    cv2.ellipse(base, (140, 140), (45, 74), 0, 0, 360, (255, 255, 255), -1)
    donor = base.copy()
    for x, y in [(118, 100), (150, 177)]:
        cv2.rectangle(base, (x, y), (x+6, y+8), (0, 0, 0), -1)
    cv2.circle(base, (169, 99), 3, (235, 235, 235), -1)
    cv2.putText(donor, "Hi!", (107, 151), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 0), 2, cv2.LINE_AA)
    matrix = np.eye(2, 3)
    japanese, = interiors(base)
    english, = interiors(donor)
    with pytest.raises(ValueError, match="perimeter"):
        propose(base, donor, matrix, japanese["text"], english["text"])
    monkeypatch.setattr(detector, "features", lambda im: None)
    monkeypatch.setattr(detector, "correspondence", lambda a, b: dict(
        matrix=matrix, inliers=100, held_errors=np.zeros(20)))
    page, _ = detector.discover(base, donor)
    region, = page["regions"]
    assert region["state"] == "suggested"
    erase, ink = region["erase"], region["ink"]
    clean, gray = propose(base, donor, matrix, erase, ink, region["erase_shape"], region["ink_shape"])
    result, _ = composite(base, [(erase, ink, clean, gray)])
    expected = donor.copy()
    cv2.circle(expected, (169, 99), 3, (235, 235, 235), -1)
    np.testing.assert_array_equal(result, expected)

    # Dark colored detail in the proposed ink group still requires review.
    base[102:107, 120:124] = [130, 40, 20]
    held, _ = detector.discover(base, donor)
    assert held["regions"][0]["state"] == "held"


def test_fitted_windows_preserve_light_marks_under_official_ink():
    base = np.full((70, 90, 3), 255, np.uint8)
    donor = base.copy()
    base[25:41, 22:25] = 0
    base[32:34, 39:43] = 234
    donor[25:41, 41:45] = 128
    donor[25:41, 45:47] = 0
    erase, ink = [12, 15, 54, 50], [30, 18, 54, 50]
    erase_shape = [[12, 15], [53, 15], [53, 49], [12, 49]]
    ink_shape = [[30, 18], [53, 18], [53, 49], [30, 49]]
    matrix = np.eye(2, 3)
    windows = detector.fitted_windows(base, donor, matrix, erase, ink, erase_shape, ink_shape)
    assert windows["paper_marks"] == [[37, 30, 45, 36]]
    clean, gray = propose(base, donor, matrix, **windows)
    result, _ = composite(base, [(erase, ink, clean, gray)])
    expected = donor.copy()
    expected[32:34, 39:41] = 234
    expected[32:34, 41:43] = (234 * 128 + 127) // 255
    np.testing.assert_array_equal(result, expected)

    for mark in ([220, 220, 220], [230, 240, 250]):
        base[32:34, 39:43] = mark
        with pytest.raises(ValueError):
            detector.fitted_windows(base, donor, matrix, erase, ink, erase_shape, ink_shape)
    base[32:34, 39:60] = 234
    with pytest.raises(ValueError):
        detector.fitted_windows(base, donor, matrix, erase, ink, erase_shape, ink_shape)


def test_edge_pairs_keep_closed_ids_and_hold_clipped_strokes(monkeypatch):
    base = np.zeros((360, 400, 3), np.uint8)
    donor = base.copy()
    for image, word in ((base, "II"), (donor, "HI!")):
        for left, top in ((0, 20), (210, 190)):
            cv2.rectangle(image, (left, top), (left+130, top+105), (255, 255, 255), -1)
            cv2.putText(image, word, (left+27, top+65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    monkeypatch.setattr(detector, "features", lambda im: None)
    monkeypatch.setattr(detector, "correspondence", lambda a, b: dict(
        matrix=np.eye(2, 3), inliers=100, held_errors=np.zeros(20)))
    page, _ = detector.discover(base, donor)
    closed, edge = page["regions"]
    assert (closed["label"], closed["state"]) == ("bubble-1", "suggested")
    assert (edge["label"], edge["state"]) == ("edge-2", "suggested")
    clean, gray = propose(base, donor, np.eye(2, 3), edge["erase"], edge["ink"])
    out, _ = composite(base, [(edge["erase"], edge["ink"], clean, gray)])
    np.testing.assert_array_equal(out[:150], donor[:150])
    donor[60:66, :12] = 0
    page, _ = detector.discover(base, donor)
    assert page["regions"][0] == closed
    assert page["regions"][1]["state"] == "held"
    assert "page edge" in page["regions"][1]["reason"]
