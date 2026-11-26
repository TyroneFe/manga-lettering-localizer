"""Run an assisted chapter in bounded, independently checkpointed jobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
import time
from zipfile import BadZipFile, ZipFile

import numpy as np
from PIL import Image

from .inventory import atomic_bytes, file_hash, natural
from .pilot import contact, locations, read_page


def load_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def write_json(path, data):
    atomic_bytes(path, (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode())


def chapter_jobs(color, english, plan):
    with ZipFile(color) as archive:
        members = {}
        for name in archive.namelist():
            path = Path(name)
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                members.setdefault(path.stem, []).append(name)
    with ZipFile(english) as archive:
        donors = sorted([n for n in archive.namelist() if plan["english_selector"] in n
                         and Path(n).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}], key=natural)
    seen = set()
    story = []
    jobs = []
    for donor in donors:
        tail = donor.split(plan["english_selector"], 1)[1]
        match = re.match(r"p(\d{3})(?:-p(\d{3}))? ", tail)
        if match is None:
            raise ValueError(f"unexpected page locator in {donor}")
        key = match[0].strip()
        if key in seen:
            raise ValueError(f"duplicate English locator {key}")
        seen.add(key)
        if key in plan["front_matter"]:
            stems = plan["front_matter"][key]
            kind = "front_matter" if stems else "english_only"
        else:
            start = int(match[1])
            end = int(match[2] or match[1])
            numbers = list(range(start, end + 1))
            if len(numbers) not in (1, 2):
                raise ValueError(f"unsupported span {key}")
            story.extend(numbers)
            stems = [f'{n + plan["color_number_offset"]:03d}' for n in numbers]
            kind = "story"
        jobs.append((key, donor, kind, stems))
    if (story != list(range(plan["first_story_page"], plan["last_story_page"] + 1))
            or not set(plan["front_matter"]).issubset(seen)):
        raise ValueError("English chapter has missing or out-of-order pages")
    jobs.extend((f"jp-{stem}", None, "color_only", [stem]) for stem in plan["color_only"])
    used = set()
    for key, donor, kind, stems in jobs:
        if len(set(stems)) != len(stems) or used.intersection(stems):
            raise ValueError(f"color member listed twice at {key}")
        used.update(stems)
        if any(len(members.get(stem, [])) != 1 for stem in stems):
            raise ValueError(f"missing or ambiguous color member at {key}")
    return [(key, donor, kind, [members[stem][0] for stem in stems])
            for key, donor, kind, stems in jobs]


def run_chapter(root, study_dir, run_id, workers=2, max_new=None):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", run_id):
        raise ValueError("invalid run ID")
    if workers < 1 or workers > 4 or (max_new is not None and max_new < 0):
        raise ValueError("use 1–4 workers and a nonnegative page limit")
    plan_file = study_dir / "chapter.json"
    config_file = study_dir / "pilot.json"
    plan = load_json(plan_file)
    color, english = locations(root, load_json(config_file))
    review = root / "review" / "chapters" / run_id
    output = root / "output" / "chapters" / run_id
    source = (root / "source").resolve()
    if any(p.resolve().is_relative_to(source) for p in (review, output)):
        raise ValueError("chapter destination resolves into source")
    pilot_review = root / "review" / "pilot"
    pilot_output = root / "output" / "pilot"
    baseline = load_json(root / "validation" / "baseline-031.json")
    pilot_manifests = {}
    for path in (pilot_review / "results.json", pilot_review / "verification.json"):
        payload = path.read_bytes()
        baseline_key = (Path("MangaColorizer") / path.relative_to(root)).as_posix()
        if baseline["files"].get(baseline_key) != sha256(payload).hexdigest():
            raise ValueError(f"accepted pilot manifest differs: {path.name}")
        pilot_manifests[path.name] = json.loads(payload)
    verification = pilot_manifests["verification.json"]
    archives = {p.relative_to(root.resolve()).as_posix(): file_hash(p) for p in (color, english)}
    if any(verification["archive_sha256"].get(name) != digest for name, digest in archives.items()):
        raise ValueError("chapter sources differ from verified pilot sources")
    verified = {m["id"]: m for m in verification["masters"]}
    pilot = {}
    for row in pilot_manifests["results.json"]:
        master = verified[row["id"]]
        path = pilot_output / master["output"]
        if row["output"] != path.name or file_hash(path) != master["file_sha256"]:
            raise ValueError(f"accepted pilot master differs: {path.name}")
        pilot[(row["english"], row["japanese"])] = (row, master, path)
    identity = {
        "chapter": plan["id"],
        "chapter_plan": file_hash(plan_file),
        "pilot_config": file_hash(config_file),
        "archives": archives,
        "accepted_masters": {m["output"]: m["file_sha256"] for m in verified.values()},
        "receipt_version": 1,
    }
    jobs = chapter_jobs(color, english, plan)
    if not set(pilot).issubset({(en, jp) for _, en, _, members in jobs for jp in members}):
        raise ValueError("chapter plan omits a verified pilot pair")

    review.mkdir(parents=True, exist_ok=True)
    # A single writer owns the run. The OS releases SQLite's file lock after a crash.
    with closing(sqlite3.connect(review / "run-lock.sqlite", timeout=0)) as lock:
        lock.execute("BEGIN IMMEDIATE")
        return process_jobs(review, output, jobs, identity, pilot, color, english, workers, max_new)


def process_jobs(review, output, jobs, identity, pilot, color, english, workers, max_new):
    manifest = review / "inputs.json"
    if manifest.exists() and load_json(manifest) != identity:
        raise ValueError("chapter inputs changed; use a fresh run ID")
    write_json(manifest, identity)
    receipts = review / "pages"
    rows = {}
    pending = []
    for job in jobs:
        key, donor, kind, members = job
        receipt = receipts / f"{key}.json"
        if receipt.exists():
            try:
                row = load_json(receipt)
                expected_masters = {
                    pilot[donor, member][1]["output"]: pilot[donor, member][1]["file_sha256"]
                    for member in members if (donor, member) in pilot
                }
                saved_masters = {view["output"]: view["sha256"]
                                 for view in row["views"] if "output" in view}
                if ((row["id"], row["english"], row["kind"]) == (key, donor, kind)
                        and row["comparison"] == f"{key}.png"
                        and [view["japanese"] for view in row["views"]] == members
                        and saved_masters == expected_masters
                        and all(path.is_file() and file_hash(path) == digest for path, digest in
                                row_artifacts(row, output, review))):
                    rows[key] = row
                    continue
            except (OSError, KeyError, TypeError, ValueError):
                pass
        pending.append(job)

    def process(job):
        key, donor, kind, members = job
        row = dict(id=key, english=donor, kind=kind, views=[], issues=[])
        donor_image = read_page(english, donor)[0] if donor else None
        panels = [("Official English", donor_image)] if donor_image is not None else []
        for member in members:
            base, _ = read_page(color, member)
            panels.append((f"Japanese color {Path(member).stem}", base))
            entry = dict(japanese=member, native_size=list(base.shape[1::-1]))
            accepted = pilot.get((donor, member))
            if accepted:
                result, master, path = accepted
                payload = path.read_bytes()
                dest = output / path.name
                atomic_bytes(dest, payload)
                if file_hash(dest) != master["file_sha256"]:
                    raise ValueError(f"staged master hash differs: {dest.name}")
                entry.update(state=result["state"], output=dest.name,
                             sha256=master["file_sha256"], retained=result["retained"])
                with Image.open(dest) as image:
                    panels.append((f"Accepted pilot {result['id']}", np.array(image)))
            else:
                entry["state"] = "needs_lettering_review" if kind == "story" else "edition_front_matter"
            row["views"].append(entry)
        if kind != "story":
            row["issues"].append("Edition front matter requires an inclusion decision")
        elif any(v["state"] == "needs_lettering_review" for v in row["views"]):
            row["issues"].append("Check page alignment and mark official English lettering regions")
        if any(v.get("retained") for v in row["views"]):
            row["issues"].append("Pilot includes retained Japanese lettering")
        comparison = review / f"{key}.png"
        contact(comparison, panels, width=360)
        row.update(comparison=comparison.name, comparison_sha256=file_hash(comparison))
        write_json(receipts / f"{key}.json", row)
        return row

    work = pending[:max_new]
    started = time.monotonic()
    original_count = len(rows)
    errors = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Only one wave is queued at a time so an interrupt leaves bounded work in flight.
        wave = {}
        try:
            for start in range(0, len(work), workers):
                wave = {job[0]: pool.submit(process, job) for job in work[start:start + workers]}
                for key, future in wave.items():
                    try:
                        rows[key] = future.result()
                    except (BadZipFile, OSError, ValueError) as exc:
                        errors[key] = str(exc)
                        print(f"{key} · {exc}", flush=True)
                    elapsed = time.monotonic() - started
                    attempted = len(rows) - original_count + len(errors)
                    eta = elapsed / attempted * (len(jobs) - len(rows))
                    print(f"{len(rows)}/{len(jobs)} {key} · estimated {eta:.0f}s remaining", flush=True)
        except KeyboardInterrupt:
            pool.shutdown(cancel_futures=True)
            for key, future in wave.items():
                if not future.cancelled() and future.exception() is None:
                    rows[key] = future.result()
    ordered = [rows[job[0]] for job in jobs if job[0] in rows]
    report = dict(chapter=identity["chapter"], complete=len(ordered) == len(jobs),
                  total_jobs=len(jobs), pending=[job[0] for job in jobs if job[0] not in rows],
                  pages=ordered, errors=errors)
    write_json(review / "report.json", report)
    write_review_index(review, report)
    return report


def row_artifacts(row, output, review):
    yield review / row["comparison"], row["comparison_sha256"]
    for view in row["views"]:
        if "output" in view:
            yield output / view["output"], view["sha256"]


def write_review_index(review, report):
    lines = [f"# Chapter review · {report['chapter']}", "",
             f"Prepared {len(report['pages'])} of {report['total_jobs']} review items.", "",
             "| Page | Disposition | Review |", "| --- | --- | --- |"]
    for row in report["pages"]:
        disposition = ". ".join(row["issues"]) or "Accepted pilot lettering"
        lines.append(f"| {row['id']} | {disposition} | [Comparison]({row['comparison']}) |")
    for key in report["pending"]:
        disposition = report["errors"].get(key, "Pending")
        lines.append(f"| {key} | {disposition.replace('|', '/')} | |")
    atomic_bytes(review / "REVIEW.md", ("\n".join(lines) + "\n").encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", type=Path)
    parser.add_argument("--run", required=True)
    parser.add_argument("--root", type=Path, default=Path("MangaColorizer"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-new", type=int)
    args = parser.parse_args()
    try:
        result = run_chapter(args.root, args.study_dir, args.run, args.workers, args.max_new)
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"{len(result['pages'])}/{result['total_jobs']} checkpointed")
    raise SystemExit(bool(result["errors"]))
