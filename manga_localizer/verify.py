"""Independently check pilot masters against archived originals and frozen geometry."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image

from .effects import read_effect_mask
from .inventory import atomic_bytes, file_hash
from .paper import read_stroke_mask
from .pilot import locations


def verify(root, config_path, annotations_path, check_sources=True, *, review=None, output=None, study_dir=None):
    config=json.loads(config_path.read_text(encoding="utf8"))
    annotations=json.loads(annotations_path.read_text(encoding="utf8"))
    review=review if review is not None else root/"review"/"pilot"
    output=output if output is not None else root/"output"/"pilot"
    jp,_=locations(root,config)
    reports=json.loads((review/"results.json").read_text(encoding="utf8"))
    if set(annotations)!={r["id"] for r in reports} or len(annotations)!=len(reports):
        raise ValueError("page accounting differs from study")
    snapshot={}
    if check_sources:
        inv=json.loads((root/"cache"/"inventory.json").read_text(encoding="utf8"))
        for edition in inv["editions"]:
            directory=Path(edition["root"])
            for rec in edition["files"]:
                digest=file_hash(directory/rec["source"])
                if digest!=rec["sha256"]:
                    raise ValueError("source hash changed: "+rec["source"])
                snapshot[(directory/rec["source"]).relative_to(root.resolve()).as_posix()]=digest
    masters=[]
    with ZipFile(jp) as originals:
        for rep in reports:
            key=rep["id"]
            filename=rep["output"]
            plan=annotations[key]
            if Path(rep["japanese"]).name!=plan["expected_color_member"]:
                raise ValueError("master corresponds to unexpected page")
            with originals.open(rep["japanese"]) as f, Image.open(f) as im:
                reference=np.asarray(im.convert("RGB"))
                icc=im.info.get("icc_profile")
            master=output/filename
            with Image.open(master) as im:
                if im.format!="PNG" or im.mode!="RGB" or im.info.get("icc_profile")!=icc:
                    raise ValueError("master encoding changed")
                decoded=np.array(im)
            if reference.shape!=decoded.shape: raise ValueError("size mismatch")
            approved=np.zeros(reference.shape[:2],np.uint8)
            held=np.zeros_like(approved)
            if [r["label"] for r in plan["regions"]]!=[r["label"] for r in rep["regions"]]:
                raise ValueError("unaccounted region")
            for region,decision in zip(plan["regions"],rep["regions"]):
                if decision["status"] not in {"applied","held"}: raise ValueError("invalid status")
                canvas=approved if decision["status"]=="applied" else held
                if "effect" in region:
                    flags=read_effect_mask(study_dir or annotations_path.parent,region["effect"],reference.shape)
                    canvas[flags!=0]=1
                    continue
                for name in ("ink","erase"):
                    x,y,r,b=region[name]
                    window=np.ones((b-y,r-x),np.uint8)
                    if name+"_shape" in region:
                        window.fill(0)
                        vertices=np.array(region[name+"_shape"],np.int32)
                        cv2.fillPoly(window,[(vertices-[x,y]).astype(np.int32)],1)
                    if name == "erase":
                        if "stroke_mask" in region:
                            window[:] = read_stroke_mask(study_dir or annotations_path.parent, region)
                        for detail in region.get("boundary_glyphs", ()):
                            vertices = (np.asarray(detail["shape"]) - [x, y]).astype(np.int32)
                            if (vertices < 0).any() or (vertices >= [r-x,b-y]).any():
                                raise ValueError("boundary glyph outside approved erase window")
                            cv2.fillPoly(window,[vertices],1)
                        for left,top,right,bottom in region.get("paper_marks", ()):
                            l,t,rr,bb=max(left,x),max(top,y),min(right,r),min(bottom,b)
                            if l<rr and t<bb:
                                window[t-y:bb-y,l-x:rr-x]=0
                    canvas[y:b,x:r] |= window
            with Image.open(review/f"{key}-authorization.png") as im:
                if not np.array_equal(np.asarray(im)>0,approved.astype(bool)):
                    raise ValueError("saved authorization disagrees with independent plan")
            delta=np.abs(decoded.astype(np.int16)-reference)
            changed=delta.max(axis=2)>0
            count=int(changed.sum())
            if (changed & ((approved==0)|(held!=0))).any() or count!=rep["changed"]:
                raise ValueError("master changed protected samples or disagrees with its report")
            if hashlib.sha256(decoded.tobytes()).hexdigest()!=rep["decoded_output_sha256"]:
                raise ValueError("decoded master digest disagrees with its report")
            masters.append(dict(id=key,output=filename,file_sha256=file_hash(master),changed=count,
                         outside_changed=0,held_changed=0,dimensions=list(decoded.shape[1::-1])))
    checks=dict(archive_sha256=snapshot,masters=masters,checked_sources=check_sources)
    atomic_bytes(review/"verification.json",json.dumps(checks,indent=2,ensure_ascii=False).encode())
    return checks


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("config",type=Path)
    p.add_argument("annotations",type=Path)
    p.add_argument("--root",type=Path,default=Path("MangaColorizer"))
    a=p.parse_args()
    v=verify(a.root,a.config,a.annotations)
    print("Verified archives:",len(v["archive_sha256"]), "masters:",len(v["masters"]))
