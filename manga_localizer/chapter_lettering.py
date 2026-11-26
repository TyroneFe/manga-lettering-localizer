"""Render an explicit selection of reviewed chapter lettering candidates."""
import argparse
import os
from pathlib import Path
import re

from .chapter import load_json, write_json
from .inventory import atomic_bytes
from .reading_index import check_index
from .render import render
from .verify import verify


def write_lettering_index(root, chapter_review, output, reports):
    """Resolve the current master for each color canvas, including split English spreads."""
    directory = chapter_review / "lettering"
    chapter = load_json(chapter_review / "report.json")
    pilot_review = root / "review" / "pilot"
    pilot = load_json(pilot_review / "results.json")
    entries = {}
    for rows, image_dir, diagnostic_dir in (
            (pilot, root / "output" / "pilot", pilot_review),
            (reports, output, directory / "rendered")):
        for row in rows:
            pair = row["english"], row["japanese"]
            entries[pair] = (row, image_dir / row["output"],
                             diagnostic_dir / f"{row['id']}-comparison.png")
    lines = []
    ready = total = 0
    for job in chapter["pages"]:
        if job["kind"] != "story":
            continue
        for view in job["views"]:
            total += 1
            key = job["id"]
            member = view["japanese"]
            entry = entries.get((job["english"], member))
            if entry is None:
                lines.append(f"| {key} | {Path(member).stem} | Awaiting lettering | [Sources](../{key}.png) |")
                continue
            row, master, comparison = entry
            ready += 1
            applied = sum(r["status"] == "applied" for r in row["regions"])
            state = "Art only" if row["state"] == "art_only" else f"{applied} transferred"
            details = [*row["retained"], *(r["label"] + " held" for r in row["regions"]
                                            if r["status"] == "held")]
            if details:
                state += ". " + " ".join(details).replace("|", "/")
            links = []
            for label, path in (("Master", master), ("Comparison", comparison)):
                if not path.is_file():
                    raise ValueError(f"missing chapter review artifact {path.name}")
                relative = Path(os.path.relpath(path, directory)).as_posix()
                links.append(f"[{label}](<{relative}>)")
            lines.append(f"| {key} | {Path(member).stem} | {state} | {' · '.join(links)} |")
    header = [f"# Lettering review · {chapter['chapter']}", "",
              f"{ready} of {total} story canvases have draft PNGs. Several pages retain Japanese lettering.",
              "Each row links to the current master. Page numbers follow the English edition and the original Japanese color members.",
              "", "[Native comparison crops](rendered/crops/) · [Chapter verification](rendered/verification.json) · [Pilot verification](../../../pilot/verification.json)",
              "", "| English | Japanese | Lettering | Files |", "| --- | --- | --- | --- |"]
    atomic_bytes(directory / "LETTERING.md", ("\n".join(header + lines) + "\n").encode())


def render_selection(root, study, run_id, selection_path):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", run_id):
        raise ValueError("invalid run ID")
    chapter_review = root / "review" / "chapters" / run_id
    review = chapter_review / "lettering" / "rendered"
    output = root / "output" / "chapters" / run_id / "lettering"
    config_path = study / "pilot.json"
    candidates = {page["id"]: page for page in load_json(chapter_review / "lettering" / "candidates.json")}
    annotations = {}
    registrations = []
    for key, selection in load_json(selection_path).items():
        if "pilot_scene" in selection:
            # A chapter revision of a retained pilot canvas has no transferred groups to replay.
            pilot_review = root / "review" / "pilot"
            scene = selection["pilot_scene"]
            prior = next(row for row in load_json(pilot_review / "results.json") if row["id"] == scene)
            if prior["changed"] or prior["regions"]:
                raise ValueError(f"{scene} already contains pilot lettering")
            registration = next(row for row in load_json(pilot_review / "registration.json") if row["id"] == scene)
            chapter = next(row for row in load_json(chapter_review / "report.json")["pages"] if row["id"] == key)
            if (chapter["english"] != registration["english"]
                    or registration["member"] not in [v["japanese"] for v in chapter["views"]]):
                raise ValueError(f"pilot and chapter disagree on {key}")
            page = dict(english_member=registration["english"], japanese_member=registration["member"],
                        matrix=registration["matrix"], regions=[])
        else:
            page = candidates[key]
        by_label = {region["label"]: region for region in page["regions"]}
        approved = []
        retained = selection["retained"]
        for label in selection["accept"]:
            region = by_label[label]
            if region["state"] != "suggested":
                raise ValueError(f"{key}/{label} did not pass the lettering preflight")
            approved.append({field: region[field] for field in
                             ("label", "erase", "ink", "erase_shape", "ink_shape", "paper_marks")
                             if field in region})
        approved.extend(selection.get("manual", []))
        if len({r["label"] for r in approved}) != len(approved):
            raise ValueError(f"{key} repeats a region label")
        if not approved and not selection.get("art_only"):
            raise ValueError(f"{key} has no approved regions or art-only disposition")
        if selection.get("art_only") and (approved or retained):
            raise ValueError(f"{key} has an inconsistent art-only disposition")
        if any(not re.fullmatch(r"[a-z0-9][a-z0-9-]*", r["label"]) for r in approved):
            raise ValueError(f"{key} has an invalid region label")
        annotations[key] = dict(expected_color_member=Path(page["japanese_member"]).name,
                                regions=approved, retained=retained)
        if selection.get("art_only"):
            annotations[key]["art_only"] = True
        if "detail_crops" in selection:
            annotations[key]["detail_crops"] = selection["detail_crops"]
        registrations.append(dict(id=key, english=page["english_member"],
                                  member=page["japanese_member"], matrix=page["matrix"]))
    source = (root / "source").resolve()
    if any(p.resolve().is_relative_to(source) for p in (review, output)):
        raise ValueError("lettering destination resolves into source")
    write_json(review / "registration.json", registrations)
    ann_path = review / "annotations.json"
    write_json(ann_path, annotations)
    reports = render(root, load_json(config_path), annotations, study_dir=study, review=review, output=output)
    write_lettering_index(root, chapter_review, output, reports)
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--run", default="gantz-ch01")
    parser.add_argument("--root", type=Path, default=Path("MangaColorizer"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--check-index", action="store_true", help="check the saved reading index and hashes read-only")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.run):
        parser.error("invalid run ID")
    if args.check_index:
        pages, groups, spreads = check_index(args.root, args.study, args.run)
        print(f"Verified {pages} indexed masters and comparisons. Transferred groups = {groups}, spreads = {spreads}")
    elif args.verify:
        review = args.root / "review" / "chapters" / args.run / "lettering" / "rendered"
        output = args.root / "output" / "chapters" / args.run / "lettering"
        checked = verify(args.root, args.study / "pilot.json", review / "annotations.json",
                         review=review, output=output, study_dir=args.study)
        print(f"Verified {len(checked['masters'])} masters and {len(checked['archive_sha256'])} archives")
    else:
        render_selection(args.root, args.study, args.run, args.study / "chapter-lettering.json")
