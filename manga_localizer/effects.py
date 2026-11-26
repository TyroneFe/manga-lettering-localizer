"""Transfer reviewed outlined lettering over artwork using a frozen two-channel mask."""
from hashlib import sha256
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


def read_effect_mask(directory, recipe, shape):
    """Bit 0 permits original-effect removal. Bit 1 permits official donor ink."""
    path = (directory / recipe["mask"]).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("effect mask outside study")
    payload = path.read_bytes()
    if sha256(payload).hexdigest() != recipe["mask_sha256"]:
        raise ValueError("reviewed effect mask hash differs")
    with Image.open(BytesIO(payload)) as im:
        if im.format != "PNG" or im.mode != "L" or im.size != shape[1::-1]:
            raise ValueError("effect mask representation differs")
        flags = np.array(im)
    if flags.max() > 3 or not flags.any():
        raise ValueError("invalid effect mask flags")
    return flags


def component_at(stats, origin):
    # Distinct clipped components may share an origin at a page corner.
    if len(origin) not in (2, 4):
        raise ValueError(f"expected a component origin or bounding box, got {origin}")
    matches = np.flatnonzero(np.all(stats[1:, :len(origin)] == origin, axis=1)) + 1
    if len(matches) != 1:
        raise ValueError(f"ambiguous component at {origin}")
    return matches[0]


def white_field_matte(gray, recipe):
    """Keep official black coverage, enclosed white faces and transparent counters."""
    dark = gray < 128
    _, labels, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8))
    selected = np.isin(labels, [component_at(stats, box) for box in recipe["ink_components"]])
    if not selected.any():
        raise ValueError("empty donor ink selection")
    _, fringes = cv2.connectedComponents((gray < 255).astype(np.uint8))
    ink = np.isin(fringes, np.unique(fringes[selected]))
    outlines, _ = cv2.findContours(selected.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    enclosed = np.zeros_like(gray)
    cv2.drawContours(enclosed, outlines, -1, 1, -1)
    ink |= (enclosed != 0) & (gray < 255)
    if (ink[[0, -1]].any() or ink[:, [0, -1]].any() or (ink & dark & ~selected).any()):
        raise ValueError("donor ink touches unrelated detail or crop edge")
    _, spaces, stats, _ = cv2.connectedComponentsWithStats((~dark).astype(np.uint8))
    faces = np.isin(spaces, [component_at(stats, box) for box in recipe["faces"]])
    if faces[[0, -1]].any() or faces[:, [0, -1]].any():
        raise ValueError("donor letter face is open")
    opaque = faces.copy()
    if faces.any():
        rim = cv2.dilate(faces.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & dark
        if np.any(rim & ~selected):
            raise ValueError("letter face belongs to unselected ink")
        # The outer edge and holes blend as black ink on the color page. The
        # inner stroke edge blends into its white letter face.
        outside = ~dark & ~faces
        to_face = cv2.distanceTransform((~faces).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        to_outside = cv2.distanceTransform((~outside).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        opaque |= ink & (to_face < to_outside)
    alpha = np.where(opaque, 1., np.where(ink, (255-gray) / 255., 0.)).astype(np.float32)
    return alpha, np.where(opaque, gray, 0).astype(np.float32)


def outlined_matte(gray, recipe):
    _, dark, stats, _ = cv2.connectedComponentsWithStats((gray < 128).astype(np.uint8))
    selected = [recipe["outline"], *recipe.get("additional_outlines", ())]
    outline = np.isin(dark, [component_at(stats, origin) for origin in selected])
    if recipe.get("face_dividers"):
        boundary = outline.astype(np.uint8)
        for start, end in recipe["face_dividers"]:
            if any(not (0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]) or not outline[y, x]
                   for x, y in (start, end)):
                raise ValueError("face divider must terminate on selected outline")
            cv2.line(boundary, start, end, 1, 1)
        outline = boundary.astype(bool)
    _, spaces, space_stats, _ = cv2.connectedComponentsWithStats((~outline).astype(np.uint8))
    faces = [component_at(space_stats, origin) for origin in recipe["faces"]]
    body = (outline | np.isin(spaces, faces)).astype(np.uint8)
    for origin in recipe["counters"]:
        rim = (dark == component_at(stats, origin)).astype(np.uint8)
        contours, hierarchy = cv2.findContours(rim, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        holes = [c for c, parent in zip(contours, hierarchy[0])
                 if parent[3] >= 0 and cv2.contourArea(c) > recipe.get("min_counter_area", 2000)]
        if len(holes) != 1:
            raise ValueError(f"ambiguous enclosed letter opening at {origin}")
        cv2.drawContours(body, holes, 0, 0, -1)

    distance = cv2.distanceTransform(1 - body, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    alpha = np.clip(recipe["keyline_radius"] + 0.5 - distance, 0, 1)
    ink = np.where(distance <= 1.5, gray, 255)
    return alpha, alpha * ink


def donor_matte(donor, matrix, recipe, size):
    width, height = recipe["donor_window"]
    x, y = recipe.get("donor_origin", [0, 0])
    gray = cv2.cvtColor(donor[y:y+height, x:x+width], cv2.COLOR_RGB2GRAY)
    if recipe.get("matte") == "white_field":
        alpha, premultiplied = white_field_matte(gray, recipe)
    elif "matte" not in recipe:
        alpha, premultiplied = outlined_matte(gray, recipe)
    else:
        raise ValueError("unknown effect matte")
    # Integrate premultiplied intensity and alpha together over each target sample.
    high = matrix.astype(np.float64).copy() * 3
    high[:, 2] += high[:, :2] @ [x, y]
    high[:, 2] += 1
    def integrate(channel):
        sampled = cv2.warpAffine(channel, high, (size[0] * 3, size[1] * 3),
                                 flags=cv2.INTER_LINEAR, borderValue=0)
        return cv2.resize(sampled, size, interpolation=cv2.INTER_AREA)

    return integrate(alpha), integrate(premultiplied)


def guide_features(donor, matrix, alpha):
    size = alpha.shape[1::-1]
    gray = cv2.cvtColor(cv2.warpAffine(donor, matrix, size), cv2.COLOR_RGB2GRAY)
    valid = (alpha < .01).astype(np.float32)
    gray = gray.astype(np.float32)
    blurred = []
    for sigma in (1.1, 3.5, 9):
        weights = cv2.GaussianBlur(valid, (0, 0), sigma)
        total = cv2.GaussianBlur(gray * valid, (0, 0), sigma)
        blurred.append(total / np.maximum(weights, .00001))
    return np.stack([np.where(valid > 0, gray, blurred[0]), 1.5 * blurred[0],
                     1.6 * blurred[1], .7 * blurred[2]], axis=2)


def guided_colors(color, features, usable, target, bounds, radius=65):
    x, y, w, h = bounds
    window = np.s_[max(0, y-radius):min(color.shape[0], y+h+radius),
                   max(0, x-radius):min(color.shape[1], x+w+radius)]
    local = usable[window]
    yy, xx = np.indices(local.shape, dtype=np.float32)
    samples = np.dstack([features[window], xx * .42, yy * .42])
    if local.sum() < 7:
        raise ValueError("insufficient clean color near effect")
    distances, indices = cv2.batchDistance(samples[target[window]], samples[local],
                                            cv2.CV_32F, K=7, normType=cv2.NORM_L2)
    weights = 1 / np.maximum(1, distances)**2
    values = (color[window][local][indices] * weights[:, :, None]).sum(axis=1)
    return (values / weights.sum(axis=1)[:, None]).round().astype(np.uint8)


def directional_samples(color, erased, target, step, limit, bounds=None):
    """Interpolate between clean Japanese samples along a measured line or texture repeat."""
    left, top, right, bottom = bounds or (0, 0, color.shape[1], color.shape[0])
    if not (0 <= left < right <= color.shape[1] and 0 <= top < bottom <= color.shape[0]):
        raise ValueError("sample bounds outside target strip")
    if limit < 1 or len(step) != 2 or not np.isfinite(step).all() or max(map(abs, step)) < 1:
        raise ValueError("invalid directional sampling step")
    yy, xx = np.nonzero(target)
    distances, anchors = [], []
    for sign in (-1, 1):
        dist = np.zeros(len(xx), dtype=np.int32)
        values = np.zeros((len(xx), 3), dtype=np.float32)
        for n in range(1, limit + 1):
            sx = xx + round(n * sign * step[0])
            sy = yy + round(n * sign * step[1])
            inside = ((sx >= left) & (sx < right) & (sy >= top) & (sy < bottom)
                      & (dist == 0))
            hits = np.flatnonzero(inside)
            hits = hits[~erased[sy[hits], sx[hits]]]
            values[hits] = color[sy[hits], sx[hits]]
            dist[hits] = n
            if dist.all():
                break
        distances.append(dist)
        anchors.append(values)
    before, after = distances
    if np.any((before + after) == 0):
        raise ValueError("no visible artwork along repair direction")
    mix = before / np.maximum(1, before + after)
    mix[before == 0] = 1
    mix[after == 0] = 0
    return np.rint(anchors[0] * (1 - mix[:, None]) + anchors[1] * mix[:, None]).astype(np.uint8)


def _bilinear_rgb(color, x, y, sx, sy):
    dx, dy = (x-sx)[:, None], (y-sy)[:, None]
    upper = (1-dx)*color[sy,sx] + dx*color[sy,sx+1]
    lower = (1-dx)*color[sy+1,sx] + dx*color[sy+1,sx+1]
    return (1-dy)*upper + dy*lower


def radial_samples(color, erased, target, center, limit, bounds):
    """Interpolate along converging lines between visible samples of the same surface."""
    l, t, r, b = bounds
    if not (0 <= l < r <= color.shape[1] and 0 <= t < b <= color.shape[0]):
        raise ValueError("ray sample window outside strip")
    if np.shape(center) != (2,) or not np.isfinite(center).all() or limit < 1:
        raise ValueError("invalid ray sampling geometry")
    yy, xx = np.nonzero(target)
    vx, vy = xx - center[0], yy - center[1]
    distance = np.hypot(vx, vy)
    if (distance < 1).any() or not np.all((xx >= l) & (xx < r) & (yy >= t) & (yy < b)):
        raise ValueError("ray target outside reviewed geometry")
    vx, vy = vx / distance, vy / distance
    total = np.zeros((len(xx), 3), np.float64)
    weights = np.zeros(len(xx), np.float64)
    for sign in (-1, 1):
        pending = np.arange(len(xx))
        for step in range(1, limit + 1):
            x = xx[pending] + sign * step * vx[pending]
            y = yy[pending] + sign * step * vy[pending]
            inside = (x >= l) & (y >= t) & (x < r-1) & (y < b-1)
            if sign < 0:
                inside &= step < distance[pending]
            ids, x, y = pending[inside], x[inside], y[inside]
            sx, sy = x.astype(int), y.astype(int)
            clean = ~(erased[sy,sx] | erased[sy,sx+1] | erased[sy+1,sx] | erased[sy+1,sx+1])
            total[ids[clean]] += _bilinear_rgb(color, x[clean], y[clean], sx[clean], sy[clean]) / step
            weights[ids[clean]] += 1 / step
            pending = ids[~clean]
            if not len(pending):
                break
    if np.any(weights == 0):
        raise ValueError("no visible color along reviewed ray")
    return np.rint(total / weights[:, None]).clip(0, 255).astype(np.uint8)


def projective_samples(color, erased, target, homography):
    """Borrow a repeated planar feature from visible artwork in this color strip."""
    transform = np.asarray(homography, dtype=np.float64)
    if transform.shape != (3, 3) or not np.isfinite(transform).all():
        raise ValueError("invalid repair homography")
    y, x = np.nonzero(target)
    points = transform @ np.array([x, y, np.ones(len(x))])
    if np.any(np.abs(points[2]) < 1e-10):
        raise ValueError("repair homography crosses infinity")
    u, v = points[:2] / points[2]
    h, w = color.shape[:2]
    if not ((u >= 0) & (u < w-1) & (v >= 0) & (v < h-1)).all():
        raise ValueError("projected repair outside color strip")
    sx, sy = u.astype(int), v.astype(int)
    if (erased[sy,sx] | erased[sy,sx+1] | erased[sy+1,sx] | erased[sy+1,sx+1]).any():
        raise ValueError("projected repair intersects erased artwork")
    return np.rint(_bilinear_rgb(color, u, v, sx, sy)).astype(np.uint8)


def gradient_samples(color, erased, target, strips):
    """Interpolate between native color strips and check the visible pixels between them."""
    l, left_end, right_start, r = strips
    if not 0 <= l <= left_end-4 < left_end < right_start <= r-4 < r <= color.shape[1]:
        raise ValueError("invalid gradient sample strips")
    rows = np.flatnonzero(erased.any(axis=1))
    if (not len(rows) or erased[:, :left_end].any() or erased[:, right_start:].any()
            or np.any(target & ~erased)):
        raise ValueError("gradient repair outside bracketing strips")
    left = color[rows, l:left_end]
    right = color[rows, right_start:r]
    if max(np.ptp(left, axis=1).max(), np.ptp(right, axis=1).max()) > 8:
        raise ValueError("gradient sample strips contain detail")
    start = np.median(left, axis=1)
    end = np.median(right, axis=1)
    xc = (l + left_end - 1) / 2
    delta = (right_start + r - 1) / 2 - xc
    fraction = (np.arange(left_end, right_start) - xc) / delta
    field = np.rint(start[:, None] + (end-start)[:, None] * fraction[None, :, None]).astype(np.uint8)
    visible = ~erased[rows, left_end:right_start]
    if (visible.sum(axis=1) < 8).any():
        raise ValueError("insufficient visible gradient on a repair row")
    error = np.abs(field.astype(np.int16) - color[rows, left_end:right_start]).max(axis=2)[visible]
    if error.max() > 10 or np.percentile(error, 99) > 4 or error.mean() > 1.5:
        raise ValueError("visible artwork disagrees with gradient repair")
    return field[target[rows, left_end:right_start]]


def propose_effect(base, donor, matrix, recipe, flags):
    """Return the composited strip, plus the number of background pixels estimated."""
    if sha256(base.tobytes()).hexdigest() != recipe["color_sha256"] or (
            sha256(donor.tobytes()).hexdigest() != recipe["donor_sha256"]):
        raise ValueError("effect recipe requires its reviewed source decodes")
    top = recipe.get("top", 0)
    height = recipe["height"]
    if not 0 <= top < top + height <= base.shape[0]:
        raise ValueError("effect strip outside color page")
    strip = slice(top, top + height)
    color = base[strip]
    placement = matrix.copy()
    placement[:, 2] += recipe.get("ink_offset", [0, 0])
    scale = recipe.get("ink_scale", 1.)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid donor lettering scale")
    placement *= scale
    placement[1, 2] -= top
    matrix = matrix.copy()
    matrix[1, 2] -= top
    alpha, ink = donor_matte(donor, placement, recipe, color.shape[1::-1])
    if (np.any(flags[:top]) or np.any(flags[strip.stop:])
            or not np.array_equal(alpha > 0, (flags[strip] & 2) != 0)):
        raise ValueError("registered donor disagrees with the frozen effect footprint")

    erased = (flags[strip] & 1) != 0
    exposed = erased & (alpha < 1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(exposed.astype(np.uint8))
    features = None
    repaired = color.copy()
    handled = set()
    if "gradient_strips" in recipe:
        repaired[exposed] = gradient_samples(color, erased, exposed, recipe["gradient_strips"])
        handled.update(range(1, count))
    for patch in recipe["repairs"]:
        index = component_at(stats, patch["origin"])
        if index in handled:
            raise ValueError("duplicate effect repair")
        handled.add(index)
        target = labels == index
        if "offset" in patch:
            dx, dy = patch["offset"]
            yy, xx = np.nonzero(target)
            sy, sx = yy + dy, xx + dx
            if (sy.min() < 0 or sx.min() < 0 or sy.max() >= height or
                    sx.max() >= color.shape[1] or erased[sy, sx].any()):
                raise ValueError("repair exemplar intersects erased or unavailable art")
            repaired[yy, xx] = color[sy, sx]
        elif "sample_homography" in patch:
            repaired[target] = projective_samples(color, erased, target, patch["sample_homography"])
        elif "texture_tile" in patch:
            x, y, w, h = patch["texture_tile"]
            if (min(x, y) < 0 or min(w, h) < 1 or x+w > color.shape[1] or y+h > height
                    or erased[y:y+h, x:x+w].any()):
                raise ValueError("texture tile intersects erased or unavailable art")
            dx, dy = patch["tile_phase"]
            yy, xx = np.nonzero(target)
            phase_x = xx + dx
            if "mirror_slope" in patch:
                # Reflect at the sampled panel joint and follow its measured slope.
                phase_x = (phase_x + np.rint(yy * patch["mirror_slope"]).astype(int)) % max(1, 2*w-2)
                phase_x = np.minimum(phase_x, 2*w-2-phase_x)
            repaired[yy, xx] = color[y + (yy+dy) % h, x + phase_x % w]
        elif "sample_step" in patch:
            repaired[target] = directional_samples(color, erased, target, patch["sample_step"],
                                                   patch["max_steps"], patch.get("sample_bounds"))
        else:
            if "ray_center" in patch:
                bounds = patch["sample_bounds"]
                left, start, right, end = bounds
                ray_target = np.zeros_like(target)
                window = np.s_[start:end, left:right]
                ray_target[window] = target[window]
                if not ray_target.any():
                    raise ValueError("empty ray repair window")
                repaired[ray_target] = radial_samples(color, erased, ray_target, patch["ray_center"],
                                                       patch["max_steps"], bounds)
                target &= ~ray_target
                if not target.any():
                    continue
            if features is None:
                # The underlying linework stays at the archive registration if only the type moves.
                guide_alpha = (alpha if np.array_equal(placement, matrix) else
                               donor_matte(donor, matrix, recipe, color.shape[1::-1])[0])
                features = guide_features(donor, matrix, guide_alpha)
                usable = ~erased & (guide_alpha < .001)
            repaired[target] = guided_colors(color, features, usable, target, stats[index, :4],
                                              recipe.get("guide_radius", 65))
    if len(handled) != count - 1:
        raise ValueError("unreviewed exposed effect component")
    result = (ink[:, :, None] + (1 - alpha[:, :, None]) * repaired).round()
    return result.clip(0, 255).astype(np.uint8), int(exposed.sum())
