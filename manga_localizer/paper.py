"""Measure near-white paper around reviewed lettering and repair the bounded stroke area."""
from hashlib import sha256
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from .lettering import (english_ink, fill_window, intersect_window, polygon_mask, rectangle,
                        validate_paper_marks, validate_white)


def smooth_paper(patch, samples, required, sigma):
    """Predict local RGB using only visible source paper within the reviewed selection."""
    def smooth(values):
        return cv2.GaussianBlur(values, (0, 0), sigma, borderType=cv2.BORDER_CONSTANT)

    weights = smooth(samples.astype(np.float32))
    if not required.any() or weights[required].min() < .015:
        raise ValueError("insufficient nearby paper samples")
    weighted = smooth(patch.astype(np.float32) * samples[:, :, None])
    prediction = weighted / np.maximum(weights[:, :, None], 1e-6)
    return np.rint(prediction).clip(0, 255).astype(np.uint8)


def measured_paper(patch, support, origin):
    if not support.any() or np.ptp(patch[support], axis=1).max() > 12:
        raise ValueError("chromatic detail in sampled paper window")
    inner = cv2.erode(support.astype(np.uint8), np.ones((5, 5), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    ring = patch[support & ~inner]
    if not len(ring) or ring.min() < 240 or np.percentile(ring, 5) < 248:
        raise ValueError("sampled paper window touches artwork or ink")
    core = (patch.min(axis=2) < 185) & support
    if not .004 < core.sum() / support.sum() < .65:
        raise ValueError("unexpected ink density on sampled paper")
    halo = cv2.dilate(core.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool) & support
    samples = support & ~halo
    if samples.sum() < 60 or patch[samples].min() < 244 or np.ptp(patch[samples], axis=1).max() > 6:
        raise ValueError("insufficient clean near-neutral paper")
    return fit_paper(patch, samples, halo, origin, sigma=12)


def fit_paper(patch, samples, halo, origin, sigma):
    # Withhold complete 8 by 8 blocks in both directions. Nearby pixels on the
    # same JPEG block cannot then provide both the fit and its validation.
    yy, xx = np.indices(samples.shape)
    fold = ((xx + origin[0]) // 8 + (yy + origin[1]) // 8) % 2 == 0
    errors = []
    for training in (fold, ~fold):
        held = samples & ~training
        if held.sum() < 25:
            raise ValueError("sparse sampled paper holdout")
        prediction = smooth_paper(patch, samples & training, held, sigma)
        errors.append(np.abs(prediction[held].astype(np.int16) - patch[held]).max(axis=1))
    error = np.concatenate(errors)
    p99 = float(np.percentile(error, 99))
    if error.max() > 8 or p99 > 4 or error.mean() > 1.5:
        raise ValueError("sampled paper fails independent block checks")
    field = smooth_paper(patch, samples, halo, sigma)
    clean = halo & ((field.astype(np.int16) - patch).max(axis=2) > p99)
    metrics = dict(samples=int(error.size), p99=round(p99, 3), worst=int(error.max()),
                   mean_absolute=round(float(error.mean()), 3))
    return clean, field, metrics


def read_stroke_mask(directory, region):
    spec = region["stroke_mask"]
    path = (directory / spec["file"]).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("stroke mask outside study")
    encoded = path.read_bytes()
    if sha256(encoded).hexdigest() != spec["sha256"]:
        raise ValueError("stroke mask hash differs")
    x, y, r, b = region["erase"]
    with Image.open(BytesIO(encoded)) as im:
        if im.format != "PNG" or im.mode != "L" or im.size != (r-x, b-y):
            raise ValueError("stroke mask representation differs")
        values = np.array(im)
    if not values.any() or np.any((values != 0) & (values != 255)):
        raise ValueError("invalid stroke mask samples")
    return values != 0


def distance_from(group):
    return cv2.distanceTransform((~group).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)


def segmented_paper(patch, mask, erase, marks):
    """Validate complete original strokes and sample paper around their actual contours."""
    value = patch.min(axis=2)
    _, labels = cv2.connectedComponents((value < 225).astype(np.uint8))
    selected_ids = np.unique(labels[mask & (value < 185)])
    if not len(selected_ids):
        raise ValueError("no original stroke in reviewed mask")
    selected = np.isin(labels, selected_ids)
    if ((selected & ~mask).any() or selected[[0, -1]].any() or selected[:, [0, -1]].any()
            or np.ptp(patch[mask], axis=1).max() > 12):
        raise ValueError("stroke mask clips original ink or includes colored detail")
    distance = distance_from(selected & (value < 185))
    clearance = distance_from((labels != 0) & ~selected)
    if np.any(mask & ((distance > 8) | (distance + .5 >= clearance))):
        raise ValueError("stroke mask reaches unrelated artwork")

    nearby = (clearance > 5) & (distance <= 24)
    _, domains = cv2.connectedComponents((nearby | mask).astype(np.uint8))
    samples = nearby & (distance > 8) & np.isin(domains, np.unique(domains[mask]))
    keep = polygon_mask(erase, exclusions=marks)
    if (mask & ~keep).any():
        raise ValueError("stroke mask overlaps a protected paper mark")
    samples &= keep
    if (samples.sum() < 60 or patch[samples].min() < 240 or np.percentile(patch[samples], 1) < 246
            or np.ptp(patch[samples], axis=1).max() > 8 or distance_from(samples)[mask].max() > 40):
        raise ValueError("insufficient intact paper around original strokes")
    return fit_paper(patch, samples, mask, erase[:2], sigma=16)


def sampled_proposal(base, donor, matrix, region, stroke_mask=None):
    marks = region.get("paper_marks", ())
    if "paper_samples" in region or (marks and stroke_mask is None):
        raise ValueError("conflicting paper specifications")
    erase, ink = region["erase"], region["ink"]
    original = base[rectangle(erase, base.shape)]
    under = base[rectangle(ink, base.shape)].copy()
    if stroke_mask is None:
        support = polygon_mask(erase, region.get("erase_shape"))
        clean, field, metrics = measured_paper(original, support, erase[:2])
    else:
        if "erase_shape" in region:
            raise ValueError("conflicting stroke mask and erase polygon")
        support = stroke_mask
        validate_paper_marks(base, erase, ink, marks)
        clean, field, metrics = segmented_paper(original, support, erase, marks)
    glyphs = english_ink(donor, matrix, ink)
    ink_support = polygon_mask(ink, region.get("ink_shape"))
    validate_white(glyphs, ink_support)
    glyphs[~ink_support] = 255
    footprint = cv2.dilate((glyphs.min(axis=2) < 255).astype(np.uint8),
                           np.ones((5, 5), np.uint8)).astype(bool)
    overlap = intersect_window(ink, erase)
    if overlap is not None:
        under[overlap][support[intersect_window(erase, ink)]] = 255
    for mark in marks:
        overlap = intersect_window(ink, mark)
        if overlap is not None:
            under[overlap] = 255
    paper = under[footprint]
    if paper.min() < 244 or np.percentile(paper, 5) < 249 or np.ptp(paper, axis=1).max() > 6:
        raise ValueError("artwork beneath the sampled paper donor")
    return (erase, ink, clean, cv2.cvtColor(glyphs, cv2.COLOR_RGB2GRAY), field), metrics


def boundary_glyph(base, donor, registration, erase, detail):
    """Validate a frozen Japanese stroke selection next to a visible balloon outline."""
    vertices = np.asarray(detail["shape"], dtype=np.int32)
    mask = polygon_mask(erase, vertices)
    left, top, width, height = cv2.boundingRect(vertices)
    # Include nearby lettering and artwork when checking the isolation of the stroke.
    box = [max(0, left-10), max(0, top-10), min(base.shape[1], left+width+10),
           min(base.shape[0], top+height+10)]
    patch = base[rectangle(box, base.shape)]
    support = polygon_mask(box, vertices)
    x, y = detail["seed"]
    if not box[0] <= x < box[2] or not box[1] <= y < box[3]:
        raise ValueError("boundary glyph seed outside reviewed window")
    _, labels, stats, _ = cv2.connectedComponentsWithStats((patch.min(axis=2) < 238).astype(np.uint8))
    label = labels[y-box[1], x-box[0]]
    if label == 0:
        raise ValueError("boundary glyph seed misses stroke")
    u, v, w, h, _ = stats[label]
    if u == 0 or v == 0 or u+w == patch.shape[1] or v+h == patch.shape[0]:
        raise ValueError("boundary glyph is connected to unreviewed artwork")
    stroke = labels == label
    other = (labels != 0) & ~stroke
    distance = distance_from(stroke)
    clearance = distance_from(other)
    if not support[stroke].all() or (support & ((distance > 3) | (distance + .5 >= clearance))).any():
        raise ValueError("boundary glyph clips a stroke or approaches protected artwork")
    paper = (distance > 3) & (distance <= 6) & (clearance > 3)
    sample = patch[paper]
    if (len(sample) < 25 or sample.min() < 244 or np.percentile(sample, 5) < 248
            or np.ptp(sample, axis=1).max() > 6 or np.ptp(patch[support], axis=1).max() > 12):
        raise ValueError("insufficient clean paper around boundary glyph")
    if english_ink(donor, registration, box)[support].min() < 250:
        raise ValueError("registered donor is not clear beneath boundary glyph")
    fill = np.rint(np.median(sample, axis=0)).astype(np.uint8)
    return mask, fill


def include_boundary_glyphs(base, donor, registration, region, proposal):
    erase, ink, clean, gray, *background = proposal
    fill = (fill_window(background[0], base, erase).copy() if background else
            np.full_like(base[rectangle(erase, base.shape)], 255))
    extra = np.zeros_like(clean)
    repair = clean.copy()
    native = base[rectangle(erase, base.shape)].astype(np.int16)
    permitted = polygon_mask(erase, exclusions=region.get("paper_marks", ()))
    for detail in region["boundary_glyphs"]:
        support, tone = boundary_glyph(base, donor, registration, erase, detail)
        if (support & (~permitted | extra)).any():
            raise ValueError("boundary glyph overlaps a preserved mark or another glyph")
        extra |= support
        fill[support] = tone
        repair |= support & ((tone.astype(np.int16) - native).max(axis=2) > 1)
    return (erase, ink, repair, gray, fill), extra
