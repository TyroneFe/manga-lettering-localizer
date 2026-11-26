"""Inventory images and archives without extracting into source directories."""
import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path, PureWindowsPath
from time import sleep
from zipfile import ZipFile

import rarfile
from PIL import Image

IMAGES = {".png", ".jpg", ".jpeg", ".webp"}
ARCHIVES = {".cbz", ".zip", ".cbr", ".rar"}
MAX_MEMBER = 128 * 1024**2
MAX_PIXELS = 80_000_000


def natural(value):
    return tuple(int(part) if i % 2 else part.casefold()
                 for i, part in enumerate(re.split(r"(\d+)", str(value))))


def safe_member(name):
    p = PureWindowsPath(name)
    return bool(p.parts) and not (p.drive or p.root or ".." in p.parts or
        any(":" in s or s.endswith((" ", ".")) or PureWindowsPath(s).is_reserved() for s in p.parts))


def file_hash(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        try:
            f.write(data)
        except BaseException:
            f.close()
            tmp.unlink(missing_ok=True)
            raise
    try:
        # A Windows reader can briefly hold the old file without delete sharing.
        for attempt in range(5):
            try:
                os.replace(tmp, path)
                break
            except PermissionError as error:
                if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 4:
                    raise
                sleep(0.05 * 2**attempt)
    finally:
        tmp.unlink(missing_ok=True)


def read_image(data, thumbnail):
    with Image.open(BytesIO(data)) as im:
        w, h = im.size
        if w*h > MAX_PIXELS:
            raise ValueError("decoded pixel limit exceeded")
        fmt, mode, info = im.format, im.mode, im.info.copy()
        depth = data[24] if fmt == "PNG" else (im.bits if fmt == "JPEG" else 8)
        icc = info.get("icc_profile")
        result = dict(format=fmt, width=w, height=h, mode=mode, bit_depth=depth,
                      frames=getattr(im, "n_frames", 1), orientation=im.getexif().get(274, 1),
                      transparency=("transparency" in info or "A" in im.getbands()),
                      icc_sha256=hashlib.sha256(icc).hexdigest() if icc else None,
                      color_tags={k:info[k] for k in ("gamma","srgb","chromaticity") if k in info})
        for i in range(result["frames"]):
            im.seek(i)
            im.load()
        if thumbnail:
            im.seek(0)
            gray = im.convert("L").resize((40, 56), Image.Resampling.BOX)
            result["proxy"] = base64.b64encode(gray.tobytes()).decode("ascii")
        return result


def scan_file(path, root, thumbnail=False, extract=None):
    label = path.relative_to(root).as_posix()
    rec = dict(source=label, bytes=path.stat().st_size, pages=[], issues=[])
    is_rar = path.suffix.lower() in {".rar", ".cbr"}
    try:
        rec["sha256"] = file_hash(path)
        if path.suffix.lower() not in ARCHIVES:
            if path.suffix.lower() not in IMAGES:
                rec["issues"].append("unsupported source")
                return rec
            members = [(0, path.name, path.stat().st_size, False, False, path)]
            container = None
        else:
            container = rarfile.RarFile(path) if is_rar else ZipFile(path)
            members = []
            for i, m in enumerate(container.infolist()):
                name = m.filename
                directory = m.isdir() if is_rar else m.is_dir()
                link = bool(m.is_symlink() or getattr(m,"file_redir",None)) if is_rar else stat.S_ISLNK(m.external_attr >> 16)
                members.append((i, name, m.file_size, directory, link, m))
        try:
            for i,name,size,directory,link,handle in sorted(members,key=lambda m:(natural(m[1]),m[1],m[0])):
                row = dict(ordinal=i, member=name, bytes=size)
                rec["pages"].append(row)
                if not safe_member(name) or link:
                    row["status"] = "unsafe member"
                elif directory or Path(name).suffix.lower() not in IMAGES:
                    row["status"] = "directory" if directory else "non-image"
                elif size > MAX_MEMBER:
                    row["status"] = "member limit exceeded"
                else:
                    try:
                        with (container.open(handle) if container else path.open("rb")) as f:
                            data = f.read(MAX_MEMBER+1)
                        if len(data) != size or len(data)>MAX_MEMBER:
                            raise ValueError("member length mismatch or limit exceeded")
                        row["sha256"] = hashlib.sha256(data).hexdigest()
                        row.update(read_image(data, thumbnail))
                        if extract:
                            dst = extract / rec["sha256"] / f"{i:06}-{row['sha256']}{Path(name).suffix.lower()}"
                            if not dst.exists() or file_hash(dst) != row["sha256"]:
                                try:
                                    atomic_bytes(dst, data)
                                except PermissionError:
                                    # A parallel scan can publish the same content first.
                                    if not dst.is_file() or file_hash(dst) != row["sha256"]:
                                        raise
                        row["status"] = "ok"
                    except Exception as exc:
                        row.update(status="unreadable", reason=f"{type(exc).__name__}: {exc}")
        finally:
            if container is not None:
                container.close()
        names = defaultdict(list)
        for entry in rec["pages"]:
            names[entry["member"].casefold()].append(entry["ordinal"])
        rec["name_collisions"] = {k:v for k,v in names.items() if len(v)>1}
    except Exception as exc:
        rec["issues"].append(f"{type(exc).__name__}: {exc}")
    return rec


def possible_gaps(records):
    groups = defaultdict(set)
    for rec in records:
        for r in rec["pages"]:
            if r["status"] != "ok":
                continue
            member = Path(r["member"])
            match = re.fullmatch(r"(.*?)(\d+)", member.stem)
            if match:
                parent = Path(rec["source"]) if Path(rec["source"]).suffix.lower() in ARCHIVES else Path(rec["source"]).parent
                key = (parent / member.parent / match[1]).as_posix()
                groups[key].add(int(match[2]))
    return {key:[[a+1,b-1] for a,b in zip(sorted(nums),sorted(nums)[1:]) if b>a+1]
            for key,nums in groups.items() if len(nums)>1 and max(nums)-min(nums)+1>len(nums)}


def inventory(inputs, report, thumbnail=False, extract=None, workers=2):
    inputs = [Path(p).resolve(strict=True) for p in inputs]
    report = Path(report).resolve()
    extract = Path(extract).resolve() if extract else None
    for p in inputs:
        for out in filter(None,(report,extract)):
            if out == p or out.is_relative_to(p) or (extract == out and p.is_relative_to(out)):
                raise ValueError("destination overlaps input")
    unrar = shutil.which("UnRAR") or Path(os.environ.get("ProgramFiles","C:/Program Files")) / "WinRAR" / "UnRAR.exe"
    if Path(unrar).is_file():
        rarfile.UNRAR_TOOL = str(unrar)
    editions = []
    for p in inputs:
        files = sorted((v for v in p.rglob("*") if v.is_file()),key=lambda v:(natural(v),str(v))) if p.is_dir() else [p]
        root = p if p.is_dir() else p.parent
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(lambda f:scan_file(f,root,thumbnail,extract),files))
        by_hash = defaultdict(list)
        for src in records:
            for row in src["pages"]:
                if row.get("sha256"):
                    by_hash[row["sha256"]].append([src["source"],row["ordinal"]])
        editions.append(dict(root=str(root), files=records, possible_gaps=possible_gaps(records),
                             duplicates={h:v for h,v in by_hash.items() if len(v)>1}))
    result = dict(schema=1,editions=editions)
    atomic_bytes(report, json.dumps(result,ensure_ascii=False,separators=(",",":")).encode())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+",type=Path)
    parser.add_argument("--report",required=True,type=Path)
    parser.add_argument("--extract",type=Path)
    parser.add_argument("--proxies",action="store_true")
    parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args()
    inventory(args.inputs,args.report,args.proxies,args.extract,args.workers)
