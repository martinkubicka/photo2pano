"""Orientation (and optionally FoV) of one photo from its rendered panorama.

    python predict.py --photo photo.jpg --pano panorama.jpg --stage1 stage1.pth [--stage2 stage2.pth]
    python predict.py --photo photo.jpg --pano panorama.jpg --stage1 stage1_fov.pth --fov-head

The photo is letterboxed to 512x512 px and the panorama to a 4096x2048 px canvas if needed (black
padding, preserved aspect ratio). Prints yaw, pitch and roll in degrees (and the horizontal FoV), with the
confidence of each stage (mean resultant length of the yaw distribution).
"""
import argparse
import io

import numpy as np
import torch
from PIL import Image

from evaluate import load, predict
from photo2pano.angles import angle_centers
from photo2pano.data import TRANSFORM
from photo2pano.padding import black_patch_mask
from photo2pano.window import content_extent_x, crop_window, window_to_real
from preprocessing.common import PANO_SIZE, QUERY_SIZE, letterbox

Image.MAX_IMAGE_PIXELS = None


def as_buffer(img, fmt="JPEG"):
    buf = io.BytesIO()
    img.save(buf, fmt, quality=100, subsampling=0) if fmt == "JPEG" else img.save(buf, fmt)
    buf.seek(0)

    return buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", required=True)
    ap.add_argument("--pano", required=True)
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", default=None)
    ap.add_argument("--fov-head", action="store_true", help="stage-1 checkpoint with FoV head")
    ap.add_argument("--crop-deg", type=float, default=180.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    centers = angle_centers(device)

    photo = Image.open(args.photo).convert("RGB")
    if photo.size != QUERY_SIZE:
        photo = letterbox(photo, QUERY_SIZE)
    pano = Image.open(args.pano).convert("RGB")
    if pano.size != PANO_SIZE:
        pano = letterbox(pano, PANO_SIZE)
    query = TRANSFORM(photo).unsqueeze(0).to(device)
    mask = black_patch_mask(photo).unsqueeze(0).to(device)

    m1 = load(args.stage1, device, args.fov_head)
    (pitch, yaw, roll), conf1, fov = predict(m1, query, as_buffer(pano, "BMP"), mask, centers)
    print(f"stage 1: yaw {yaw:.2f}, pitch {pitch:.2f}, roll {roll:.2f}, confidence {conf1:.3f}"
          + (f", FoV {fov:.2f}" if args.fov_head else ""))

    if args.stage2:
        del m1
        m2 = load(args.stage2, device)
        pano_np = np.asarray(pano)
        min_x, max_x = content_extent_x(pano_np)
        width_ratio = (max_x - min_x + 1) / float(pano_np.shape[1])
        win = Image.fromarray(crop_window(pano_np, yaw, args.crop_deg, content_x=(min_x, max_x)))
        aw, conf2, _ = predict(m2, query, as_buffer(win), mask, centers)
        pitch, yaw, roll = window_to_real(yaw, aw[0], aw[1], aw[2], args.crop_deg, width_ratio)
        print(f"stage 2: yaw {yaw:.2f}, pitch {pitch:.2f}, roll {roll:.2f}, confidence {conf2:.3f} "
              f"(combined confidence {conf1 * conf2:.3f})")


if __name__ == "__main__":
    main()
