"""Check a saved chapter reading index against its plan and both master manifests."""
from pathlib import Path
import re

from PIL import Image

from .chapter import load_json
from .inventory import file_hash


def _masters(review, output):
    reports = load_json(review / "results.json")
    seals = load_json(review / "verification.json")["masters"]
    by_id = {row["id"]: row for row in seals}
    if len(by_id) != len(seals) or len(reports) != len(seals) or {r["id"] for r in reports} != set(by_id):
        raise ValueError(f"results and verification disagree in {review}")
    entries = {}
    for row in reports:
        seal = by_id[row["id"]]
        name = row["output"]
        if (name != f"{Path(row['japanese']).stem}-{row['id']}.png" or seal["output"] != name
                or not re.fullmatch(r"[A-Za-z0-9-]+", row["id"])
                or seal["outside_changed"] or seal["held_changed"] or row["changed"] != seal["changed"]):
            raise ValueError(f"invalid master record for {name}")
        pair = row["english"], row["japanese"]
        if pair in entries:
            raise ValueError(f"duplicate edition pair for {name}")
        entries[pair] = (row, seal, output / name, review / f"{row['id']}-comparison.png")
    return entries


def check_index(root, study, run_id):
    plan = load_json(study / "chapter.json")
    directory = root / "review" / "chapters" / run_id / "lettering"
    report = load_json(directory.parent / "report.json")
    if (not report["complete"] or report["errors"] or report["pending"]
            or report["total_jobs"] != len(report["pages"]) or report["chapter"] != plan["id"]):
        raise ValueError("chapter report is incomplete or belongs to a different study")
    entries = _masters(root / "review" / "pilot", root / "output" / "pilot")
    entries.update(_masters(directory / "rendered", root / "output" / "chapters" / run_id / "lettering"))
    # Compare the actual links with the verified file for each edition pair.
    rows = [line for line in (directory / "LETTERING.md").read_text(encoding="utf8").splitlines()
            if line.startswith("|") and not line.startswith(("| English |", "| --- |"))]
    pattern = re.compile(r"\| ([^|]+) \| ([^|]+) \| ([^|]+) \| \[Master\]\(<([^>]+)>\) · \[Comparison\]\(<([^>]+)>\) \|")
    position = total = spreads = 0
    seen = set()
    for job in report["pages"]:
        if job["kind"] != "story":
            continue
        views = job["views"]
        first = plan["first_story_page"] + position
        locator = f"p{first:03d}"
        if len(views) == 2:
            locator += f"-p{first+1:03d}"
            spreads += 1
        if len(views) not in (1, 2) or job["id"] != locator:
            raise ValueError(f"unexpected story locator or spread at {job['id']}")
        for view in views:
            pair = job["english"], view["japanese"]
            stem = f"{plan['first_story_page'] + position + plan['color_number_offset']:03d}"
            match = pattern.fullmatch(rows[position]) if position < len(rows) else None
            if (match is None or match.groups()[:2] != (locator, stem) or Path(pair[1]).stem != stem
                    or pair not in entries or pair[1] in seen):
                raise ValueError(f"missing, duplicated or out-of-order canvas at {locator}/{stem}")
            seen.add(pair[1])
            row, seal, master, comparison = entries[pair]
            count = sum(r["status"] == "applied" for r in row["regions"])
            status = "Art only" if row["state"] == "art_only" else f"{count} transferred"
            if (match[3].split(". ", 1)[0] != status
                    or (directory / match[4]).resolve() != master.resolve()
                    or (directory / match[5]).resolve() != comparison.resolve()):
                raise ValueError(f"index status or links differ at {locator}/{stem}")
            if file_hash(master) != seal["file_sha256"]:
                raise ValueError(f"master hash differs for {master.name}")
            with Image.open(comparison) as image:
                if image.format != "PNG":
                    raise ValueError(f"comparison is not a PNG for {locator}/{stem}")
                image.verify()
            position += 1
            total += count
    if len(rows) != position or position != plan["last_story_page"] - plan["first_story_page"] + 1:
        raise ValueError("reading index does not cover the planned story range")
    return position, total, spreads
