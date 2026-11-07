"""Zoom-in window of the second stage.

A `crop_deg` wide window of the panorama content, centered at a yaw and at the vertical center of the
canvas, is resized to the canvas size and processed like a full panorama. Angles inside the window are
magnified by scale = 360 / crop_deg (2x for 180 deg); pitch additionally by 1 / width_ratio for
panoramas whose content does not span the full canvas width.
"""
import numpy as np
from PIL import Image

from .angles import wrap

CANVAS_HW = (2048, 4096)


def content_extent_x(pano, threshold=16):
    """First and last column of the panorama canvas that contain image content."""
    col = np.asarray(pano).max(axis=(0, 2))
    xs = np.where(col > threshold)[0]
    if len(xs) == 0:
        return 0, pano.shape[1] - 1

    return int(xs.min()), int(xs.max())


def yaw_to_x(yaw_deg, content_w, min_x=0.0):
    """yaw -180 -> left edge of the content, +180 -> right edge."""
    return min_x + (wrap(yaw_deg) + 180.0) / 360.0 * content_w


def crop_window(pano, center_yaw_deg, crop_deg=180.0, out_hw=CANVAS_HW, content_x=None):
    """Window of the panorama content centered at center_yaw_deg, wrapping around the 360 deg seam."""
    pano = np.asarray(pano)
    H, W = pano.shape[:2]
    out_h, out_w = out_hw
    min_x, max_x = (0, W - 1) if content_x is None else content_x
    content_w = int(max_x) - int(min_x) + 1

    crop_w = max(1, min(content_w, int(round(content_w * float(crop_deg) / 360.0))))
    crop_h = max(1, min(H, int(round(crop_w * out_h / out_w))))
    x0 = int(round(yaw_to_x(center_yaw_deg, content_w, min_x) - crop_w / 2.0))
    y0 = max(0, min(H - crop_h, int(round(H / 2.0 - crop_h / 2.0))))

    rel = (x0 - min_x) % content_w
    if rel + crop_w <= content_w:
        crop = pano[y0:y0 + crop_h, min_x + rel:min_x + rel + crop_w]
    else:
        crop = np.concatenate([pano[y0:y0 + crop_h, min_x + rel:min_x + content_w],
                               pano[y0:y0 + crop_h, min_x:min_x + rel + crop_w - content_w]], axis=1)

    if crop.shape[:2] != (out_h, out_w):
        crop = np.array(Image.fromarray(crop).resize((out_w, out_h), Image.LANCZOS))

    return crop


def scales(crop_deg, width_ratio=1.0):
    yaw_scale = 360.0 / float(crop_deg)

    return yaw_scale, yaw_scale / float(width_ratio)


def real_to_window(center_yaw_deg, pitch, yaw, roll, crop_deg=180.0, flipped=False, width_ratio=1.0):
    """Labels in the frame of a window centered at center_yaw_deg."""
    ys, ps = scales(crop_deg, width_ratio)
    sign = -1.0 if flipped else 1.0

    return pitch * ps, wrap(wrap(yaw - center_yaw_deg) * ys * sign), wrap(sign * roll)


def window_to_real(center_yaw_deg, pitch_w, yaw_w, roll_w, crop_deg=180.0, width_ratio=1.0):
    """Second-stage prediction (window frame) -> absolute (pitch, yaw, roll)."""
    ys, ps = scales(crop_deg, width_ratio)

    return wrap(pitch_w / ps, 180.0), wrap(center_yaw_deg + yaw_w / ys), wrap(roll_w)
