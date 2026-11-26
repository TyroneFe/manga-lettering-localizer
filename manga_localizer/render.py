"""Render and verify the six original color canvases in the five-case review set."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .effects import propose_effect, read_effect_mask
from .inventory import atomic_bytes
from .lettering import composite, lettering_placement, polygon_mask, propose, rectangle, reviewed_paper
from .paper import include_boundary_glyphs, read_stroke_mask, sampled_proposal
from .pilot import contact, locations, read_page, save_png


def comparison_window(region, shape):
    boxes = [region["erase"], region["ink"]]
    if region["status"] == "applied" and ("placement_scale" in region or "placement_offset" in region):
        # Show all original donor lines, even if the color edition has a shorter caption box.
        adjustment = lettering_placement(np.eye(2, 3), region)
        corners = np.asarray(region["ink"]).reshape(2, 2)
        original = (corners - adjustment[:, 2]) / adjustment[0, 0]
        boxes.append([*np.floor(original[0]), *np.ceil(original[1])])
    x, y = np.min(boxes, axis=0).astype(int)[:2]
    r, b = np.max(boxes, axis=0).astype(int)[2:]
    return np.s_[max(0,y-16):min(shape[0],b+16), max(0,x-16):min(shape[1],r+16)]


def render(root,config,annotations,study_dir=None,*,review=None,output=None):
    jp,en=locations(root,config)
    review=review if review is not None else root/"review"/"pilot"
    output=output if output is not None else root/"output"/"pilot"
    source=(root/"source").resolve()
    if any(p.resolve().is_relative_to(source) for p in (review,output)):
        raise ValueError("output resolves inside sources")
    reg=json.loads((review/"registration.json").read_text(encoding="utf8"))
    if {r["id"] for r in reg}!=set(annotations):
        raise ValueError("registration and annotation sets differ")
    reports=[]
    for record in reg:
        key=record["id"]
        row=annotations[key]
        if Path(record["member"]).name!=row["expected_color_member"]:
            raise ValueError(f"{key} requires page identification review")
        base,icc=read_page(jp,record["member"])
        donor,_=read_page(en,record["english"])
        M=np.asarray(record["matrix"])
        proposals=[]
        effects=[]
        decisions=[]
        frozen=np.zeros(base.shape[:2],bool)
        for region in row["regions"]:
            erase,ink=region["erase"],region["ink"]
            if "effect" in region:
                recipe=region["effect"]
                flags=read_effect_mask(study_dir,recipe,base.shape)
                strip,reconstructed=propose_effect(base,donor,M,recipe,flags)
                footprint=flags!=0
                effects.append((strip,footprint,recipe.get("top",0)))
                decisions.append(dict(label=region["label"],erase=erase,ink=ink,
                                      kind="outlined_effect",status="applied",
                                      original_effect_area=int(np.count_nonzero(flags & 1)),
                                      reconstructed=reconstructed))
                continue
            try:
                placement=lettering_placement(M,region)
                diagnostics = {}
                strokes = None
                if "paper_method" in region or "stroke_mask" in region:
                    method = region.get("paper_method")
                    if method == "segmented" and "stroke_mask" in region:
                        strokes = read_stroke_mask(study_dir or review, region)
                    elif method != "local_fit" or "stroke_mask" in region:
                        raise ValueError("unknown paper method")
                    proposal, diagnostics["paper_holdout"] = sampled_proposal(base,donor,placement,region,strokes)
                else:
                    paper = (reviewed_paper(base,erase,ink,region["paper_samples"])
                             if "paper_samples" in region else None)
                    clean,gray=propose(base,donor,placement,erase,ink,region.get("erase_shape"),
                                       region.get("ink_shape"),paper=paper,
                                       paper_marks=region.get("paper_marks", ()))
                    proposal=(erase,ink,clean,gray)
                    if paper is not None:
                        proposal=(*proposal,paper)
                detail_support = None
                if region.get("boundary_glyphs"):
                    proposal, detail_support = include_boundary_glyphs(base,donor,M,region,proposal)
                proposals.append(proposal)
                for name,box in (("erase",erase),("ink",ink)):
                    exclusions = region.get("paper_marks", ()) if name == "erase" else ()
                    support = strokes if name == "erase" and strokes is not None else polygon_mask(
                        box,region.get(name+"_shape"),exclusions)
                    frozen[rectangle(box,base.shape)] |= support
                if detail_support is not None:
                    frozen[rectangle(erase,base.shape)] |= detail_support
                decisions.append(dict(**region,**diagnostics,status="applied"))
            except ValueError as err:
                decisions.append(dict(**region,status="held",reason=str(err)))
        result,_=composite(base,proposals)
        for strip,footprint,top in effects:
            if np.any(frozen & footprint):
                raise ValueError("overlapping reviewed effects or text regions")
            frozen |= footprint
            window=slice(top,top+len(strip))
            mask=footprint[window]
            result[window][mask]=strip[mask]
        save_png(review/f"{key}-authorization.png",frozen.astype(np.uint8)*255)
        applied=len(proposals)+len(effects)
        dest=output/f"{Path(record['member']).stem}-{key}.png"
        save_png(dest,result,icc)
        with Image.open(dest) as saved:
            if saved.mode!="RGB" or saved.size!=base.shape[1::-1] or saved.info.get("icc_profile")!=icc:
                raise AssertionError("output representation differs")
            decoded=np.array(saved)
        with Image.open(review/f"{key}-authorization.png") as saved_mask:
            allowed=np.array(saved_mask)>0
        diff=np.any(decoded!=base,axis=2)
        if np.any(diff & ~allowed) or not np.array_equal(decoded,result):
            raise AssertionError("saved master failed independent pixel check")
        registered=cv2.warpAffine(donor,M,base.shape[1::-1],borderValue=(220,220,220))
        contact(review/f"{key}-comparison.png",[("Japanese color",base),("Official English (original)",donor),
                ("Candidate / review" if applied else "Retained original / review",decoded)],width=480)
        overlay=base.copy()
        overlay[diff]=[255,0,128]
        for region in decisions:
            for box in (region["erase"],region["ink"]):
                x,y,r,b=box
                cv2.rectangle(overlay,(x,y),(r-1,b-1),(0,120,255),3)
        save_png(review/f"{key}-diff.png",overlay)
        # These crops keep one image pixel per source pixel for human inspection.
        for region in decisions:
            window=comparison_window(region,base.shape)
            strip=np.concatenate([im[window] for im in (base,registered,decoded)],axis=1)
            save_png(review/"crops"/f"{key}-{region['label']}.png",strip)
        for label,box in row.get("detail_crops",{}).items():
            window=rectangle(box,base.shape)
            strip=np.concatenate([im[window] for im in (base,registered,decoded)],axis=1)
            save_png(review/"crops"/f"{key}-{label}.png",strip)
        state = "partial_review" if applied else "original_retained_review"
        if row.get("art_only"):
            if applied or decisions or row["retained"]:
                raise ValueError(f"{key} art-only page includes lettering decisions")
            state = "art_only"
        report=dict(id=key,english=record["english"],japanese=record["member"],output=dest.name,
                    state=state,
                    native_size=list(base.shape[1::-1]),approved_window_pixels=int(allowed.sum()),
                    changed=int(diff.sum()),outside_changed=int((diff & ~allowed).sum()),
                    outside_max_channel_delta=int(np.abs(decoded.astype('int16')-base)[~allowed].max(initial=0)),
                    decoded_source_sha256=sha256(base.tobytes()).hexdigest(),
                    decoded_output_sha256=sha256(decoded.tobytes()).hexdigest(),
                    icc_sha256=sha256(icc).hexdigest() if icc else None,
                    regions=decisions,retained=row["retained"])
        reports.append(report)
        print(key,applied,"/",len(decisions),report["changed"],
              [(d["label"],d["reason"]) for d in decisions if d["status"]=="held"])
    atomic_bytes(review/"results.json",json.dumps(reports,indent=2,ensure_ascii=False).encode())
    return reports


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("config",type=Path)
    p.add_argument("annotations",type=Path)
    p.add_argument("--root",type=Path,default=Path("MangaColorizer"))
    a=p.parse_args()
    render(a.root,json.loads(a.config.read_text(encoding="utf8")),
           json.loads(a.annotations.read_text(encoding="utf8")),a.annotations.parent)
