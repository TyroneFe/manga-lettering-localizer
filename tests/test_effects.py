from hashlib import sha256

import cv2
import numpy as np
import pytest
from PIL import Image

from manga_localizer.effects import (component_at, directional_samples, donor_matte, gradient_samples,
                                    outlined_matte, projective_samples, propose_effect,
                                    radial_samples, read_effect_mask, white_field_matte)


def test_outlined_letter_preserves_counter_and_reconstructs_old_effect(tmp_path):
    # Original synthetic art and lettering. The grey counter must reveal color art.
    donor = np.full((210, 260, 3), 180, np.uint8)
    cv2.rectangle(donor, (18, 15), (170, 180), (0, 0, 0), -1)
    cv2.rectangle(donor, (27, 24), (161, 171), (255, 255, 255), -1)
    cv2.circle(donor, (94, 95), 39, (0, 0, 0), -1)
    cv2.circle(donor, (94, 95), 32, (180, 180, 180), -1)
    rgb = [37, 118, 84]
    base = np.empty_like(donor)
    base[:] = rgb
    base[70:76, 230:235] = [255, 0, 0]
    base[122:129, 200:209] = [255, 0, 0]
    original = base.copy()
    matrix = np.eye(2, 3)
    recipe = dict(donor_window=[260, 210], height=210, outline=[18, 15],
                  faces=[[27, 24]], counters=[[55, 56]], keyline_radius=6,
                  repairs=[{"origin": [230, 70], "offset": [-12, 0]}, {"origin": [200, 122]}],
                  color_sha256=sha256(base.tobytes()).hexdigest(),
                  donor_sha256=sha256(donor.tobytes()).hexdigest())
    alpha, _ = donor_matte(donor, matrix, recipe, (260, 210))
    flags = (alpha > 0).astype(np.uint8) * 2
    flags[70:76, 230:235] |= 1
    flags[122:129, 200:209] |= 1
    path = tmp_path / "mask.png"
    Image.fromarray(flags).save(path)
    recipe.update(mask=path.name, mask_sha256=sha256(path.read_bytes()).hexdigest())
    frozen = read_effect_mask(tmp_path, recipe, base.shape)
    result, count = propose_effect(base, donor, matrix, recipe, frozen)
    assert count == 30 + 63
    assert (result[70:76, 230:235] == rgb).all()
    assert (result[122:129, 200:209] == rgb).all()
    assert (result[95, 94] == rgb).all()
    assert (result[70, 30] == 255).all()
    assert (result[30, 22] == 0).all()
    assert np.array_equal(result[flags == 0], base[flags == 0])
    assert np.array_equal(base, original)

    shifted = matrix.copy()
    shifted[0, 2] += 1
    with pytest.raises(ValueError, match="frozen effect footprint"):
        propose_effect(base, donor, shifted, recipe, frozen)
    flags[2, 2] ^= 1
    Image.fromarray(flags).save(path)
    with pytest.raises(ValueError, match="mask hash"):
        read_effect_mask(tmp_path, recipe, base.shape)


@pytest.mark.parametrize("scale", [1., 1.024])
def test_cropped_donor_and_shifted_strip_match_full_page_projection(scale):
    donor = np.full((128, 148, 3), 180, np.uint8)
    cv2.rectangle(donor, (51, 68), (90, 93), (0, 0, 0), -1)
    cv2.rectangle(donor, (55, 72), (86, 89), (255, 255, 255), -1)
    base = np.full((115, 135, 3), [31, 97, 133], np.uint8)
    matrix = np.array([[.875, 0., 4.], [0., .875, 6.]])
    full = dict(donor_window=[148, 128], outline=[51, 68], faces=[[55, 72]],
                counters=[], keyline_radius=3)
    placement = matrix.copy()
    placement[1, 2] -= 3
    placement *= scale
    alpha, ink = donor_matte(donor, placement, full, base.shape[1::-1])
    recipe = dict(full, donor_origin=[32, 46], donor_window=[83, 69],
                  outline=[19, 22], faces=[[23, 26]], top=50, height=50,
                  ink_offset=[0, -3], ink_scale=scale, repairs=[],
                  color_sha256=sha256(base.tobytes()).hexdigest(),
                  donor_sha256=sha256(donor.tobytes()).hexdigest())
    flags = (alpha > 0).astype(np.uint8) * 2
    strip, count = propose_effect(base, donor, matrix, recipe, flags)
    expected = (ink[:, :, None] + (1 - alpha[:, :, None]) * base).round().astype(np.uint8)
    actual = base.copy()
    actual[50:100] = strip
    assert count == 0
    assert np.array_equal(actual, expected)
    assert np.array_equal(actual[flags == 0], base[flags == 0])
    flags[4, 17] = 1
    with pytest.raises(ValueError, match="frozen effect footprint"):
        propose_effect(base, donor, matrix, recipe, flags)


def test_divided_outline_preserves_native_white_face_at_page_edge():
    gray = np.full((50, 65), 180, np.uint8)
    gray[18:, 20:41] = 255
    gray[18:, 17:20] = 0
    gray[18:, 41:44] = 0
    recipe = dict(outline=[17, 18], additional_outlines=[[41, 18]],
                  face_dividers=[[[18, 18], [42, 18]]], faces=[[20, 19, 21, 31]],
                  counters=[], keyline_radius=2)
    alpha, ink = outlined_matte(gray, recipe)
    assert np.all(alpha[18:, 17:44] == 1)
    np.testing.assert_array_equal(ink[18:, 17:44], gray[18:, 17:44])
    assert alpha[-1, 10] == 0 and alpha[10, 30] == 0
    recipe["face_dividers"][0][1] = [40, 18]
    with pytest.raises(ValueError, match="terminate on selected outline"):
        outlined_matte(gray, recipe)


def test_clipped_components_and_directional_repair():
    stats = np.array([[0, 0, 8, 6, 20], [0, 0, 4, 5, 12], [0, 0, 6, 6, 16]])
    with pytest.raises(ValueError, match="ambiguous"):
        component_at(stats, [0, 0])
    assert component_at(stats, [0, 0, 6, 6]) == 2

    # The horizontal stripes are completely obscured in the middle. The
    # fully masked last row has no available source along this direction.
    rows, cols = np.indices((5, 11))
    original = np.repeat((rows * 20 + cols * 3)[:, :, None], 3, axis=2).astype(np.uint8)
    erased = np.zeros(rows.shape, bool)
    erased[:, 3:8] = True
    damaged = original.copy()
    damaged[erased] = 255
    assert np.array_equal(directional_samples(damaged, erased, erased, [1, 0], 8), original[erased])
    single = np.zeros_like(erased)
    single[1, 3] = True
    assert np.array_equal(directional_samples(damaged, erased, single, [1, 0], 8, [2, 0, 7, 4])[0],
                          original[1, 2])
    erased[-1, :] = True
    with pytest.raises(ValueError, match="no visible artwork"):
        directional_samples(damaged, erased, erased, [1, 0], 11)


def test_projected_bay_uses_only_clean_in_bounds_samples():
    yy, xx = np.indices((10, 11))
    rgb = np.stack((8*xx, 4*yy, 2*xx+4*yy), axis=-1).astype(np.uint8)
    target = (xx == 2) & (yy == 2)
    mask = target.copy()
    transform = np.array([[1., 0., 2.5], [0., 1., 1.25], [0., 0., 1.]])
    assert projective_samples(rgb, mask, target, transform).tolist() == [[36, 13, 22]]
    mask[4, 5] = True
    with pytest.raises(ValueError, match="intersects erased"):
        projective_samples(rgb, mask, target, transform)
    transform[0, 2] = -3
    with pytest.raises(ValueError, match="outside color strip"):
        projective_samples(rgb, mask, target, transform)


def test_bounded_rays_restore_converging_lines_from_clean_color():
    yy, xx = np.indices((90, 110))
    angle = (yy+20) / (xx+35)
    line = (angle > .55) & (angle < .68)
    base = np.stack((xx+20*line, 2*yy+35*line, xx+yy+30*line), axis=2).astype(np.uint8)
    erase = (xx >= 38) & (xx <= 55) & (yy >= 12) & (yy <= 49)
    # Intersections well inside or outside the ruled line have exact affine RGB.
    target = erase & (np.abs(angle-.55) > .05) & (np.abs(angle-.68) > .05)
    damaged = base.copy()
    damaged[erase] = [250, 45, 210]
    result = radial_samples(damaged, erase, target, [-35, -20], 48, [0, 0, 110, 90])
    np.testing.assert_array_equal(result, base[target])
    with pytest.raises(ValueError, match="no visible color"):
        radial_samples(damaged, erase, target, [-35, -20], 48, [38, 12, 56, 50])
    with pytest.raises(ValueError, match="outside reviewed geometry"):
        radial_samples(damaged, erase, target, [-35, -20], 48, [45, 0, 110, 90])


def test_white_field_outline_preserves_raster_and_transparent_counter():
    gray = np.full((55, 65), 255, np.uint8)
    gray[10:44, 12:49] = 0
    gray[10:44, 11] = 128
    gray[14:40, 16:45] = 255
    gray[22:32, 25:36] = 0
    gray[24:30, 27:34] = 255
    gray[26, 30] = 238
    recipe = dict(ink_components=[[12, 10], [25, 22]], faces=[[16, 14]])
    alpha, ink = white_field_matte(gray, recipe)
    np.testing.assert_array_equal(np.rint(ink + (1-alpha)*255).astype(np.uint8), gray)
    over_color = np.rint(ink + (1-alpha)*150).astype(np.uint8)
    assert over_color[16, 11] == round(150*128/255)
    assert over_color[15, 20] == 255 and over_color[25, 29] == 150
    assert over_color[26, 30] == round(150*238/255)
    gray[20, 3:12] = 238
    gray[20, 3] = 0
    with pytest.raises(ValueError, match="unrelated"):
        white_field_matte(gray, recipe)


def test_row_gradient_reconstruction_and_independent_visible_controls():
    yy, xx = np.indices((24, 80))
    reference = np.stack((70+xx+yy, 20+xx+2*yy, 180-xx+yy), axis=2).astype(np.uint8)
    mask = (xx >= 33) & (xx <= 46) & (yy >= 5) & (yy < 17)
    damaged = reference.copy()
    damaged[mask] = [30, 50, 244]
    exposed = mask & (xx > 36)
    np.testing.assert_array_equal(gradient_samples(damaged, mask, exposed, [12, 18, 64, 70]),
                                  reference[exposed])
    damaged[11, 27] = [25, 15, 35]
    with pytest.raises(ValueError, match="disagrees"):
        gradient_samples(damaged, mask, exposed, [12, 18, 64, 70])
    with pytest.raises(ValueError, match="bracketing"):
        gradient_samples(reference, mask, exposed, [30, 36, 64, 70])
