"""Look for surviving source ink inside chapter balloons using saved PNGs and registered donor ink."""
import argparse
from hashlib import sha256
from pathlib import Path
import re

import cv2
import numpy as np
from PIL import Image

from .chapter import load_json, write_json
from .discover import overlap
from .lettering import english_ink, lettering_placement, polygon_mask, rectangle
from .pilot import locations, read_page


def inspect_face(base, master, donor_gray, box):
    """Compare dark source samples to the actual master and expected donor coverage."""
    patch = rectangle(box, base.shape)
    source = base[patch].min(axis=2)
    result = master[patch].min(axis=2)
    expected = donor_gray[patch]
    enclosed = np.zeros_like(source, np.uint8)
    height, width = enclosed.shape
    found = False
    for threshold in (247, 241, 235, 229):
        white = (source >= threshold).astype(np.uint8)
        contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        faces = [c for c in contours if cv2.boundingRect(c) == (0, 0, width, height)]
        if len(faces) == 1:
            found = True
            cv2.drawContours(enclosed, faces, 0, 1, -1)
    if not found:
        return None
    dark = (enclosed != 0) & (source < 180) & (result < 195) & (result.astype(np.int16) + 18 < expected)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8))
    findings = []
    for label, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if area < 5:
            continue
        local = np.s_[y:y+h, x:x+w]
        digest = sha256()
        for plane in (labels[local] == label, base[patch][local], master[patch][local], expected[local]):
            digest.update(plane.tobytes())
        left, top = int(box[0]+x), int(box[1]+y)
        findings.append(dict(box=[left, top, left+int(w), top+int(h)], pixels=int(area),
                             signature=digest.hexdigest()))
    return findings


def inspect_page(base, master, donor, matrix, regions, faces):
    donor_gray = np.full(base.shape[:2], 255, np.uint8)
    associations = {}
    unmatched = []
    for g in regions:
        if g["status"] != "applied" or "effect" in g or g.get("kind") == "outlined_effect":
            continue
        ink = g["ink"]
        samples = english_ink(donor, lettering_placement(matrix, g), ink)
        coverage = cv2.cvtColor(samples, cv2.COLOR_RGB2GRAY)
        coverage[~polygon_mask(ink, g.get("ink_shape"))] = 255
        window = rectangle(ink, base.shape)
        donor_gray[window] = np.minimum(donor_gray[window], coverage)
        scores = [overlap(g["erase"], f["text"]) for f in faces]
        if not scores or max(scores) < .23:
            unmatched.append(g["label"])
            continue
        associations.setdefault(int(np.argmax(scores)), []).append(g["label"])
    findings = []
    checked = 0
    for index, groups in associations.items():
        box = faces[index]["bubble"]
        components = inspect_face(base, master, donor_gray, box)
        if components is None:
            unmatched.extend(groups)
            continue
        checked += 1
        findings.extend(dict(groups=groups, **component) for component in components)
    return dict(checked_faces=checked, unmatched=unmatched, findings=findings)


def audit(root, study, run_id):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", run_id):
        raise ValueError("invalid run ID")
    directory = root / "review" / "chapters" / run_id / "lettering"
    rendered = directory / "rendered"
    destination = (rendered / "residual-review.json").resolve()
    if (not destination.is_relative_to((root / "review").resolve())
            or destination.is_relative_to((root / "source").resolve())):
        raise ValueError("invalid residual review destination")
    output = root / "output" / "chapters" / run_id / "lettering"
    jp, en = locations(root, load_json(study / "pilot.json"))
    candidates = {r["id"]: r for r in load_json(directory / "candidates.json")}
    registrations = {r["id"]: r for r in load_json(rendered / "registration.json")}
    selections = load_json(study / "chapter-lettering.json")
    rows = []
    for page in load_json(rendered / "results.json"):
        key = page["id"]
        reg = registrations[key]
        if reg["member"] != page["japanese"] or reg["english"] != page["english"]:
            raise ValueError(f"registration does not match the saved render for {key}")
        base, _ = read_page(jp, reg["member"])
        donor, _ = read_page(en, reg["english"])
        with Image.open(output / page["output"]) as im:
            master = np.array(im)
        if (sha256(base.tobytes()).hexdigest() != page["decoded_source_sha256"]
                or sha256(master.tobytes()).hexdigest() != page["decoded_output_sha256"]):
            raise ValueError(f"decoded source or master differs from the render record for {key}")
        faces = candidates.get(key, {}).get("japanese", [])
        row = inspect_page(base, master, donor, np.asarray(reg["matrix"]), page["regions"], faces)
        reviews = selections[key].get("residual_reviews", {})
        for item in row["findings"]:
            if item["signature"] in reviews:
                item["disposition"] = reviews[item["signature"]]
        row["unused_reviews"] = sorted(set(reviews) - {item["signature"] for item in row["findings"]})
        rows.append(dict(id=key, decoded_output_sha256=page["decoded_output_sha256"], **row))
    result = dict(checked_faces=sum(row["checked_faces"] for row in rows),
                  unmatched_groups=sum(len(row["unmatched"]) for row in rows),
                  pending=sum("disposition" not in item for row in rows for item in row["findings"]),
                  pages=rows)
    write_json(destination, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--root", type=Path, default=Path("MangaColorizer"))
    parser.add_argument("--run", default="gantz-ch01")
    args = parser.parse_args()
    result = audit(args.root, args.study, args.run)
    print(f"Checked faces = {result['checked_faces']}, unmatched groups = {result['unmatched_groups']}, "
          f"pending findings = {result['pending']}")
