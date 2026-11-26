"""Transfer a monochrome glyph group within independently reviewed, flat white interiors."""
import cv2
import numpy as np


def rectangle(box, shape):
    x,y,r,b=box
    if not (0<=x<r<=shape[1] and 0<=y<b<=shape[0]):
        raise ValueError("region outside page")
    return np.s_[y:b,x:r]


def english_ink(donor, matrix, box):
    """Integrate the native donor's glyph coverage over each target pixel."""
    x,y,r,b=box
    offset=np.rint(matrix[:,2]).astype(int)
    if np.array_equal(matrix[:,:2],np.eye(2)) and np.array_equal(offset,matrix[:,2]):
        dx,dy=offset
        sl=rectangle([x-dx,y-dy,r-dx,b-dy],donor.shape)
        return donor[sl].copy()
    scale=3
    high=matrix.copy()
    high[:,:2]*=scale
    high[:,2]=(matrix[:,2]-[x,y])*scale+(scale-1)/2
    samples=cv2.warpAffine(donor,high,((r-x)*scale,(b-y)*scale),
                           flags=cv2.INTER_LINEAR,borderValue=(127,127,127))
    return cv2.resize(samples,(r-x,b-y),interpolation=cv2.INTER_AREA)


def intersect_window(box, other):
    x, y, r, b = box
    left, top, right, bottom = other
    l, t, rr, bb = max(left, x), max(top, y), min(right, r), min(bottom, b)
    if l < rr and t < bb:
        return np.s_[t-y:bb-y, l-x:rr-x]
    return None


def polygon_mask(box, vertices=None, exclusions=()):
    x,y,r,b=box
    mask=np.ones((b-y,r-x),np.uint8)
    if vertices is not None:
        pts=np.asarray(vertices,np.int32)-[x,y]
        if np.any(pts<0) or np.any(pts>=[r-x,b-y]):
            raise ValueError("polygon outside its review window")
        mask.fill(0)
        cv2.fillPoly(mask,[pts.astype(np.int32)],1)
    for exclusion in exclusions:
        overlap = intersect_window(box, exclusion)
        if overlap is not None:
            mask[overlap] = 0
    return mask.astype(bool)


def lettering_placement(matrix, region):
    """Scale a registered donor around a reviewed point on the color page, then translate."""
    placement = np.array(matrix, dtype=np.float64, copy=True)
    scale = np.asarray(region.get("placement_scale", 1.0), dtype=np.float64)
    anchor = np.asarray(region.get("placement_anchor", [0, 0]), dtype=np.float64)
    offset = np.asarray(region.get("placement_offset", [0, 0]), dtype=np.float64)
    if (scale.ndim != 0 or not np.isfinite(scale) or scale <= 0
            or anchor.shape != (2,) or offset.shape != (2,)
            or not np.isfinite(anchor).all() or not np.isfinite(offset).all()):
        raise ValueError("invalid lettering placement")
    placement *= scale
    placement[:, 2] += (1 - scale) * anchor + offset
    return placement


def validate_white(patch, support):
    if patch.ndim!=3 or patch.shape[2]!=3 or min(patch.shape[:2])<7:
        raise ValueError("invalid flat text window")
    inner=cv2.erode(support.astype(np.uint8),np.ones((5,5),np.uint8),
                    borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    ring=patch[support & ~inner]
    if not len(ring):
        raise ValueError("empty text-window perimeter")
    if ring.min()<240 or np.mean(ring==255)<.80:
        raise ValueError(f"nonwhite or clipped perimeter (min {ring.min()}, white {np.mean(ring==255):.3f})")
    if np.percentile(np.ptp(patch[support].astype(np.int16),axis=1),99.9)>12:
        raise ValueError("chromatic detail in text window")
    darkness=255-patch.min(axis=2)
    if not (.004<np.mean(darkness[support]>120)<.65):
        raise ValueError("unexpected ink density")


def reviewed_paper(base, erase, ink, samples):
    """Estimate a light row-wise background from two reviewed clean strips."""
    top, bottom = min(erase[1], ink[1]), max(erase[3], ink[3])
    if len(samples) != 2 or any(not 0 <= x < r <= base.shape[1] for x, r in samples):
        raise ValueError("invalid paper sample strips")
    strips = np.concatenate([base[top:bottom, x:r] for x, r in samples], axis=1)
    if strips.min() < 245 or np.ptp(strips, axis=1).max() > 3:
        raise ValueError("sample strips do not agree on light paper color")
    paper = np.full((base.shape[0], 1, 3), 255, np.uint8)
    paper[top:bottom, 0] = np.rint(np.median(strips, axis=1))
    return paper


def validate_paper_marks(base, erase, ink, paper_marks):
    for mark in paper_marks:
        if len(mark) != 4 or any(not isinstance(v, (int, np.integer)) for v in mark):
            raise ValueError("paper mark requires four integer coordinates")
        x, y, r, b = mark
        patch = base[rectangle(mark, base.shape)]
        if (max(r-x, b-y) > 16 or patch.min() < 224 or np.ptp(patch, axis=2).max() > 4
                or not any(l <= x and t <= y and r <= rr and b <= bb for l,t,rr,bb in (erase,ink))):
            raise ValueError("invalid reviewed paper mark")


def propose(base, donor, matrix, erase, ink, erase_shape=None, ink_shape=None, *,
            paper=None, paper_marks=()):
    es=rectangle(erase,base.shape)
    ds=rectangle(ink,base.shape)
    original=base[es]
    if paper is not None:
        original=np.minimum(original.astype(np.uint16)+255-paper[es[0]],255).astype(np.uint8)
    validate_paper_marks(base, erase, ink, paper_marks)
    erase_support=polygon_mask(erase,erase_shape,paper_marks)
    ink_support=polygon_mask(ink,ink_shape)
    validate_white(original,erase_support)
    glyphs=english_ink(donor,matrix,ink)
    validate_white(glyphs,ink_support)
    glyphs[~ink_support]=255
    # The target area underneath donor glyphs must either be authorized for
    # cleaning or already be white; its background is independently checked.
    under=base[ds].copy()
    if paper is not None:
        under=np.minimum(under.astype(np.uint16)+255-paper[ds[0]],255).astype(np.uint8)
    # These light marks stay in the base. Only genuine donor ink may overprint them.
    for mark in paper_marks:
        overlap = intersect_window(ink, mark)
        if overlap is not None:
            under[overlap] = 255
    x,y,r,b=ink
    ex,ey,er,eb=erase
    l,t,rr,bb=max(ex,x),max(ey,y),min(er,r),min(eb,b)
    if l<rr and t<bb:
        cleared=erase_support[t-ey:bb-ey,l-ex:rr-ex]
        under[t-y:bb-y,l-x:rr-x][cleared]=255
    footprint=cv2.dilate((glyphs.min(axis=2)<255).astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    if under[footprint].min()<245 or np.mean(under[footprint]==255)<.95:
        raise ValueError("destination includes art or ink outside erase support")
    # Only erase pixels close to solid neutral strokes. Include JPEG ringing.
    core=((original.min(axis=2)<185) & erase_support).astype(np.uint8)
    near=cv2.dilate(core,np.ones((17,17),np.uint8)).astype(bool)
    changed=original.min(axis=2)<255
    if np.any((original.min(axis=2)<248) & erase_support & ~near):
        raise ValueError("nonuniform background outside stroke support")
    return near & changed & erase_support, cv2.cvtColor(glyphs,cv2.COLOR_RGB2GRAY)


def fill_window(field, base, erase):
    window = rectangle(erase, base.shape)
    shape = base[window].shape
    if field.shape == shape:
        return field
    return np.broadcast_to(field[window[0]], shape)


def composite(base, proposals):
    out=base.copy()
    used=np.zeros(base.shape[:2],bool)
    for erase,ink,clean,gray,*background in proposals:
        es=rectangle(erase,base.shape)
        ds=rectangle(ink,base.shape)
        if used[es].any() or used[ds].any(): raise ValueError("overlapping groups")
        if background:
            out[es][clean] = fill_window(background[0],base,erase)[clean]
        else:
            out[es][clean] = 255
        # Reference ink is neutral on white. White donor samples have zero opacity.
        bg=out[ds].astype(np.uint16)
        out[ds]=((bg*gray[:,:,None]+127)//255).astype(np.uint8)
        used[es]=True
        used[ds]=True
    return out,used
