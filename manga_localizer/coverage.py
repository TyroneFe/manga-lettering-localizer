"""Establish chapter-level evidence from independent cross-edition artwork matches."""
import argparse
import base64
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import cv2
import numpy as np

from .inventory import atomic_bytes


def features(gray):
    # Compare local luminance structure and suppress broad painted or blank areas.
    small=cv2.resize(gray,(30,42),interpolation=cv2.INTER_AREA).astype(np.float32)
    contrast=small-cv2.GaussianBlur(small,(0,0),2)
    norm=np.sqrt(cv2.GaussianBlur(contrast**2,(0,0),2)+100)
    v=np.clip(contrast/norm,-2,2)[2:-2,1:-1].ravel()
    return v/max(1,np.linalg.norm(v))


def collect(edition):
    records=[]
    descriptors=[]
    for book,f in enumerate(edition["files"]):
        for row in f["pages"]:
            if row["status"]!="ok": continue
            page=np.frombuffer(base64.b64decode(row["proxy"]),np.uint8).reshape(56,40)
            scene=re.search(r" - c(\d+(?:#\d+)?) ",row["member"])
            projections=[("full",page)] if row["width"] < row["height"] else [("right",page[:,20:]),("left",page[:,:20])]
            for side,part in projections:
                records.append((book,row["member"],scene[1] if scene else None,side))
                descriptors.append(features(part))
    return records,np.stack(descriptors)


def run(index,dest):
    jp,en=json.loads(index.read_text(encoding="utf8"))["editions"]
    js,jv=collect(jp)
    es,ev=collect(en)
    hits=defaultdict(list)
    strongest={}
    # One ordered sequence is not assumed. Each supporting page must win in both directions.
    for begin in range(0,len(js),128):
        scores=jv[begin:begin+128] @ ev.T
        for offset,s in enumerate(scores):
            i=begin+offset
            j=int(s.argmax())
            top=float(s[j])
            s[j]=-2
            margin=top-float(s.max())
            if top<.39 or margin<.060: continue
            if top>strongest.get(j,(-1,None))[0]:
                strongest[j]=(top,(i,j,margin))
    # Check reciprocal retrieval including rivals rejected by the forward margin.
    items=list(strongest.items())
    for b in range(0,len(items),128):
        group=items[b:b+128]
        best=(ev[[j for j,_ in group]] @ jv.T).argmax(axis=1)
        for (j,(score,(i,_,margin))),rev in zip(group,best):
            if i!=rev: continue
            book,name,_,_=js[i]
            eb,en_name,ch,side=es[j]
            hits[book,ch].append(dict(japanese=name,english_archive=en["files"][eb]["source"],english=en_name,
                                       side=side,score=round(score,4),margin=round(margin,4)))
    books=[]
    for i,f in enumerate(jp["files"]):
        chapters={}
        for (book,ch),seq in hits.items():
            if book!=i: continue
            seq.sort(key=lambda q:q["japanese"])
            chapters[ch]=seq
        books.append(dict(source=f["source"],chapters=chapters))
        print(i+1,[(ch,len(v)) for ch,v in sorted(chapters.items())],flush=True)
    editions=[]
    for e in [jp,en]:
        stats=[]
        for f in e["files"]:
            rows=[r for r in f["pages"] if r["status"]=="ok"]
            specs=Counter((r["format"],r["mode"],r["width"],r["height"]) for r in rows)
            chapters=Counter(m[1] for r in rows if (m:=re.search(r" - c(\d+(?:#\d+)?) ",r["member"])))
            stats.append(dict(archive=f["source"],sha256=f["sha256"],images=len(rows),member_chapter_labels=chapters,
                              formats_and_resolutions=[[*k,v] for k,v in sorted(specs.items())]))
        editions.append(dict(edition=Path(e["root"]).name,archives=stats))
    atomic_bytes(dest,json.dumps(dict(editions=editions,japanese_chapter_evidence=books),
                                 ensure_ascii=False,indent=1).encode())


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("inventory",type=Path)
    p.add_argument("report",type=Path)
    a=p.parse_args()
    run(a.inventory,a.report)
