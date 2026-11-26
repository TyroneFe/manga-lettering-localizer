"""Suggest paired white lettering regions for operator review."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re

import cv2
import numpy as np

from .chapter import load_json, write_json
from .lettering import english_ink, polygon_mask, propose
from .pilot import contact, locations, read_page
from .registration import correspondence, features


def touches_edge(box, shape):
    x, y, right, bottom = box
    return x == 0 or y == 0 or right == shape[1] or bottom == shape[0]


def interiors(image, *, include_edges=False):
    white = (image.min(axis=2) >= 247).astype(np.uint8)
    contours, hierarchy = cv2.findContours(white, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    if hierarchy is None:
        return candidates
    height, width = white.shape
    for index, contour in enumerate(contours):
        child, parent = hierarchy[0, index, [2, 3]]
        if parent >= 0 or child < 0:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        bubble = [x, y, x+w, y+h]
        if (min(w, h) < 24 or not 750 <= area <= .16 * height * width
                or (not include_edges and touches_edge(bubble, white.shape)) or area < .45 * w * h):
            continue
        holes = []
        while child >= 0:
            hole = contours[child]
            if cv2.contourArea(hole) >= 5:
                holes.append(hole)
            child = hierarchy[0, child, 0]
        if len(holes) < 2:
            continue
        left, top, tw, th = cv2.boundingRect(np.concatenate(holes))
        if min(tw, th) < 8 or tw * th > .85 * w * h:
            continue
        text_box = [max(0, left - 8), max(0, top - 8),
                    min(width, left + tw + 8), min(height, top + th + 8)]
        candidates.append(dict(bubble=bubble, text=text_box))
    return sorted(candidates, key=lambda b: (touches_edge(b["bubble"], white.shape),
                                             b["bubble"][1], -b["bubble"][0]))


def overlap(a, b):
    x, y, r, t = a
    u, v, s, z = b
    common = max(0, min(r, s) - max(x, u)) * max(0, min(t, z) - max(y, v))
    return common / ((r-x)*(t-y) + (s-u)*(z-v) - common)


def white_face(patch):
    white = (patch >= 247).astype(np.uint8)
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounds = (0, 0, patch.shape[1], patch.shape[0])
    faces = [contour for contour in contours if cv2.boundingRect(contour) == bounds]
    if len(faces) != 1:
        raise ValueError("ambiguous white interior")
    interior = np.zeros_like(white)
    cv2.drawContours(interior, faces, 0, 1, -1)
    return interior


def glyph_window(image, bubble, padding=10):
    """Fit a polygon to dark enclosed strokes and leave a margin inside the white face."""
    x, y, right, bottom = bubble
    patch = image[y:bottom, x:right].min(axis=2)
    interior = white_face(patch)
    core = ((patch < 185) & (interior != 0)).astype(np.uint8)
    points = cv2.findNonZero(core)
    if points is None:
        raise ValueError("no enclosed dark lettering")
    hull = np.zeros_like(interior)
    cv2.fillConvexPoly(hull, cv2.convexHull(points), 1)
    size = 2 * padding + 1
    hull = cv2.dilate(hull, np.ones((size, size), np.uint8))
    safe = cv2.erode(interior, np.ones((7, 7), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)
    support = hull & safe
    if np.any(core & ~support):
        raise ValueError("lettering touches the bubble edge")
    outlines, _ = cv2.findContours(support, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(outlines) != 1:
        raise ValueError("fragmented text support")
    vertices = cv2.approxPolyDP(outlines[0], 0.7, True)
    # Simplification may cut inward near a tight corner.
    hull.fill(0)
    cv2.fillPoly(hull, [vertices], 1)
    if np.any(core & ~hull):
        raise ValueError("fitted boundary clips dark lettering")
    vertices = vertices.reshape(-1, 2) + np.int32([x, y])
    left, top, width, height = cv2.boundingRect(vertices)
    return [left, top, left+width, top+height], vertices.tolist()


def faint_paper_marks(base, donor, matrix, erase, ink, erase_shape, ink_shape):
    """Suggest small, light islands to preserve when the surrounding paper is cleaned."""
    x, y = min(erase[0], ink[0]), min(erase[1], ink[1])
    right, bottom = max(erase[2], ink[2]), max(erase[3], ink[3])
    patch = base[y:bottom, x:right]
    value = patch.min(axis=2)
    erase_mask = np.zeros(value.shape, np.uint8)
    cv2.fillPoly(erase_mask, [np.int32(erase_shape) - np.int32([x, y])], 1)
    core = ((value < 185) & (erase_mask != 0)).astype(np.uint8)
    near = cv2.dilate(core, np.ones((17, 17), np.uint8))
    relevant = (erase_mask != 0) & (near == 0) & (value < 248)
    l, t, r, b = ink
    window = np.s_[t-y:b-y, l-x:r-x]
    glyphs = (english_ink(donor, matrix, ink).min(axis=2) < 255) & polygon_mask(ink, ink_shape)
    footprint = cv2.dilate(glyphs.astype(np.uint8), np.ones((5, 5), np.uint8))
    relevant[window] |= (footprint != 0) & (erase_mask[window] == 0) & (value[window] < 245)

    # Measure the whole connected mark, including the portion outside the ink footprint.
    _, labels, stats, _ = cv2.connectedComponentsWithStats((value < 248).astype(np.uint8))
    marks = []
    for label in np.unique(labels[relevant]):
        u, v, w, h, _ = stats[label]
        if u == 0 or v == 0 or u+w == value.shape[1] or v+h == value.shape[0]:
            continue
        left, top = max(0, u-2), max(0, v-2)
        r, b = min(right-x, u+w+2), min(bottom-y, v+h+2)
        local = patch[top:b, left:r]
        box = [int(x+left), int(y+top), int(x+r), int(y+b)]
        if (max(r-left, b-top) <= 16 and local.min() >= 224 and np.ptp(local, axis=2).max() <= 4
                and any(l <= box[0] and t <= box[1] and box[2] <= rr and box[3] <= bb
                        for l, t, rr, bb in (erase, ink))):
            marks.append(box)
    return marks


def fitted_windows(base, donor, matrix, erase, ink, erase_shape, ink_shape):
    windows = dict(erase=erase, ink=ink, erase_shape=erase_shape, ink_shape=ink_shape)
    try:
        propose(base, donor, matrix, **windows)
    except ValueError:
        marks = faint_paper_marks(base, donor, matrix, **windows)
        if not marks:
            raise
        windows["paper_marks"] = marks
        propose(base, donor, matrix, **windows)
    return windows


def lettering_windows(base, donor, matrix, projected, japanese, english):
    erase, ink = japanese["text"], english["text"]
    try:
        propose(base, donor, matrix, erase, ink)
        return dict(erase=erase, ink=ink)
    except ValueError:
        pass
    ink, ink_shape = glyph_window(projected, english["bubble"])
    for padding in (10, 16):
        erase, erase_shape = glyph_window(base, japanese["bubble"], padding)
        try:
            return fitted_windows(base, donor, matrix, erase, ink, erase_shape, ink_shape)
        except ValueError:
            if padding == 16:
                raise


def check_page_edge(image, bubble):
    """Hold a face with dark interruptions along its exposed cut at the page edge."""
    if not touches_edge(bubble, image.shape):
        return
    x, y, r, b = bubble
    value = image[y:b, x:r].min(axis=2)
    face = white_face(value)
    for at_edge, cut in ((x == 0, np.s_[:, 0]), (r == image.shape[1], np.s_[:, -1]),
                         (y == 0, np.s_[0, :]), (b == image.shape[0], np.s_[-1, :])):
        if at_edge:
            line = value[cut]
            paper = np.flatnonzero((line >= 247) & (face[cut] != 0))
            if len(paper) and line[paper[0]:paper[-1]+1].min() < 185:
                raise ValueError("interrupted paper at the page edge")


def associated_pairs(japanese, english):
    links = [[overlap(j["bubble"], e["bubble"]) for e in english] for j in japanese]
    for j, scores in enumerate(links):
        if not scores:
            break
        e = int(np.argmax(scores))
        reverse = [row[e] for row in links]
        if scores[e] < .48 or int(np.argmax(reverse)) != j:
            continue
        rivals = sorted(scores)[-2:-1] + sorted(reverse)[-2:-1]
        if rivals and scores[e] - max(rivals) < .12:
            continue
        yield j, e, scores[e]


def discover(base, donor):
    registration = correspondence(features(donor), features(base))
    if registration is None or registration["inliers"] < 50:
        return {"state": "registration_review", "regions": []}, None
    matrix = registration["matrix"]
    projected = cv2.warpAffine(donor, matrix, base.shape[1::-1], borderValue=(180, 180, 180))
    japanese, english = interiors(base, include_edges=True), interiors(projected, include_edges=True)
    closed_j = [v for v in japanese if not touches_edge(v["bubble"], base.shape)]
    closed_e = [v for v in english if not touches_edge(v["bubble"], base.shape)]
    # Keep the existing internal pair IDs when an edition also has marginal balloons.
    closed_pairs = list(associated_pairs(closed_j, closed_e))
    used_j = {j for j, _, _ in closed_pairs}
    used_e = {e for _, e, _ in closed_pairs}
    edge_pairs = [(j, e, score) for j, e, score in associated_pairs(japanese, english)
                  if (j >= len(closed_j) or e >= len(closed_e)) and j not in used_j and e not in used_e]
    regions = []
    for j, e, score in closed_pairs + edge_pairs:
        edge = j >= len(closed_j) or e >= len(closed_e)
        erase, ink = japanese[j]["text"], english[e]["text"]
        row = dict(label=f"{'edge' if edge else 'bubble'}-{j+1}", erase=erase, ink=ink,
                   japanese_index=j, english_index=e, overlap=round(score, 3))
        try:
            if edge:
                check_page_edge(base, japanese[j]["bubble"])
                check_page_edge(projected, english[e]["bubble"])
            windows = lettering_windows(base, donor, matrix, projected, japanese[j], english[e])
        except ValueError as exc:
            row.update(state="held", reason=str(exc))
        else:
            row.update(state="suggested", **windows)
        regions.append(row)
    held = registration["held_errors"]
    return dict(state="lettering_review", matrix=matrix.tolist(),
                inliers=registration["inliers"], holdout_samples=len(held),
                holdout_p95=float(np.percentile(held, 95)) if len(held) else None,
                japanese=japanese, english=english, regions=regions), projected


def run(root, study, run_id):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", run_id):
        raise ValueError("invalid run ID")
    color, english = locations(root, load_json(study / "pilot.json"))
    review = root / "review" / "chapters" / run_id
    directory = review / "lettering"
    source = (root / "source").resolve()
    if not directory.resolve().is_relative_to((root / "review").resolve()) or directory.resolve().is_relative_to(source):
        raise ValueError("invalid review directory")
    report = load_json(review / "report.json")

    def inspect(item):
        key, donor_name, member = item
        base, _ = read_page(color, member)
        donor, _ = read_page(english, donor_name)
        row, projected = discover(base, donor)
        row.update(id=key, japanese_member=member, english_member=donor_name)
        if projected is not None:
            left, right = base.copy(), projected.copy()
            for region in row["regions"]:
                paint = (30, 160, 70) if region["state"] == "suggested" else (210, 70, 50)
                for image, side in ((left, "erase"), (right, "ink")):
                    x, y, r, b = region[side]
                    shape = region.get(side + "_shape")
                    if shape is None:
                        cv2.rectangle(image, (x, y), (r-1, b-1), paint, 3)
                    else:
                        cv2.polylines(image, [np.int32(shape)], True, paint, 3)
                    label = region["label"].replace("bubble-", "").replace("edge-", "E")
                    cv2.putText(image, label, (x, max(20, y-7)),
                                cv2.FONT_HERSHEY_SIMPLEX, .8, paint, 2)
                for l, t, r, b in region.get("paper_marks", []):
                    cv2.rectangle(left, (l, t), (r-1, b-1), (220, 60, 180), 1)
            contact(directory / f"{key}.png", [("Japanese color", left), ("Registered English", right)], width=600)
        write_json(directory / "pages" / f"{key}.json", row)
        return row

    jobs = [(p["id"], p["english"], v["japanese"])
            for p in report["pages"] for v in p["views"] if v["state"] == "needs_lettering_review"]
    if len({job[0] for job in jobs}) != len(jobs):
        raise ValueError("lettering discovery requires separately registered spread views")
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(inspect, jobs))
    write_json(directory / "candidates.json", rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--root", type=Path, default=Path("MangaColorizer"))
    parser.add_argument("--run", default="gantz-ch01")
    args = parser.parse_args()
    rows = run(args.root, args.study, args.run)
    print(f"Reviewed geometry for {len(rows)} pages")
    for row in rows:
        print(row["id"], len(row.get("japanese", [])), len(row.get("english", [])),
              sum(r["state"] == "suggested" for r in row["regions"]), len(row["regions"]))
