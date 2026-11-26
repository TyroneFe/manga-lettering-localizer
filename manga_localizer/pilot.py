"""Prepare a small, explicitly selected review set. All positions use target pixels."""
import argparse
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .inventory import atomic_bytes, file_hash, natural
from .registration import correspondence, features


def read_page(archive, member):
    with ZipFile(archive) as z, z.open(member) as f, Image.open(f) as im:
        if im.getexif().get(274,1) != 1:
            raise ValueError("prototype needs native upright inputs")
        if im.mode not in {"P","RGB","L"} or "transparency" in im.info:
            raise ValueError("prototype needs opaque 8-bit images")
        return np.array(im.convert("RGB")), im.info.get("icc_profile")


def save_png(path, pixels, icc=None):
    buf=BytesIO()
    Image.fromarray(pixels).save(buf, format="PNG", icc_profile=icc)
    atomic_bytes(path,buf.getvalue())


def contact(path, panels, width=420):
    views = []
    for label,pix in panels:
        im=Image.fromarray(pix)
        im.thumbnail((width, round(width*1.52)),Image.Resampling.LANCZOS)
        views.append((label,im))
    sheet=Image.new("RGB",(width*len(views),max(im.height for _,im in views)+25),"white")
    draw=ImageDraw.Draw(sheet)
    for k,(label,im) in enumerate(views):
        draw.text((k*width+5,5),label,fill="black")
        sheet.paste(im,(k*width,25))
    save_png(path,np.asarray(sheet))


def locations(root, config):
    source=(root/"source").resolve()
    jp=(source/config["color_archive"]).resolve()
    en=(source/config["english_archive"]).resolve()
    if not (jp.is_relative_to(source) and en.is_relative_to(source)):
        raise ValueError("invalid source locator")
    return jp,en


def prepare(root, config):
    jp,en = locations(root,config)
    review=root/"review"/"pilot"
    with ZipFile(jp) as z:
        candidates=sorted([i.filename for i in z.infolist() if i.filename.lower().endswith((".jpg",".png",".webp"))],key=natural)
    a,b=config["candidate_range"]
    candidates=candidates[a-1:b]
    with ZipFile(en) as z:
        names=z.namelist()
    targets={}
    for case in config["cases"]:
        hits=[n for n in names if case["english_contains"] in n]
        if len(hits)!=1:
            raise ValueError(f"donor selector hits {len(hits)} members")
        image,_=read_page(en,hits[0])
        for view in case["views"]:
            x1,x2=view["donor_fraction"]
            start,end=round(image.shape[1]*x1),round(image.shape[1]*x2)
            pts,desc=features(image[:,start:end])
            pts[:,0]+=start
            targets[view["id"]]=(hits[0],(pts,desc))
    rankings={key:[] for key in targets}
    for member in candidates:
        color,_=read_page(jp,member)
        f=features(color)
        for key, (_,donor) in targets.items():
            result=correspondence(donor,f)
            if result is None:
                continue
            errors=result["held_errors"]
            rankings[key].append(dict(member=member, inliers=result["inliers"],
                held_in_12px=len(errors), held_total=result["held_total"],
                median_px=float(np.median(errors)) if len(errors) else None,
                p95_px=float(np.percentile(errors,95)) if len(errors) else None,
                held_xy=result["held_xy"].tolist(),held_errors=errors.tolist(),matrix=result["matrix"].tolist()))
    records=[]
    for key,rank in rankings.items():
        rank.sort(key=lambda r:r["inliers"],reverse=True)
        if not rank:
            raise ValueError(f"unmatched {key}")
        best=rank[0]
        best.update(id=key,english=targets[key][0],runner_up=[(r["member"],r["inliers"]) for r in rank[1:4]])
        color,_=read_page(jp,best["member"])
        donor,_=read_page(en,best["english"])
        warped=cv2.warpAffine(donor,np.asarray(best["matrix"]),color.shape[1::-1],borderValue=(245,245,245))
        contact(review/f"{key}-alignment.png",[("Japanese original",color),("Registered official English",warped)],width=560)
        print(key,Path(best["member"]).name,best["inliers"],best["median_px"],best["p95_px"],flush=True)
        records.append(best)
    atomic_bytes(review/"registration.json",json.dumps(records,ensure_ascii=False,indent=2).encode())


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("config",type=Path)
    p.add_argument("--root",type=Path,default=Path("MangaColorizer"))
    args=p.parse_args()
    prepare(args.root,json.loads(args.config.read_text(encoding="utf8")))
