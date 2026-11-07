"""Build the training set of the second stage.

Every panorama of --in-dir is replaced by --per-image windows of --crop-deg degrees, centered at
yaw_gt + delta with delta a simulated first-stage yaw error, and the labels are written in the window
frame. The output has the usual layout and is used with train.py.

    python build_stage2_dataset.py --in-dir data/train --pano-dir data/train_8k --out-dir data/train_s2 \
        --per-image 6 --augment --workers 8
"""
import argparse
import csv
import os
import zlib
from multiprocessing import Pool

import numpy as np
from PIL import Image

from photo2pano.angles import wrap
from photo2pano.data import read_gt
from photo2pano.offsets import sample_offsets
from photo2pano.window import content_extent_x, crop_window, real_to_window
from preprocessing.augment import color_jitter_keep_black

Image.MAX_IMAGE_PIXELS = None
PITCH_LIMIT = 89.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True, help="photos, gt csv files and panoramas")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pano-dir", default=None, help="higher resolution panoramas (default: --in-dir)")
    p.add_argument("--crop-deg", type=float, default=180.0)
    p.add_argument("--out-width", type=int, default=4096)
    p.add_argument("--out-height", type=int, default=2048)
    p.add_argument("--per-image", type=int, default=6)
    p.add_argument("--offset-mode", choices=["mixture", "empirical"], default="mixture")
    p.add_argument("--errors-csv", default=None, help="csv with a yaw_err_signed column (empirical mode)")
    p.add_argument("--max-delta", type=float, default=75.0)
    p.add_argument("--augment", action="store_true", help="colour jitter of the photo and joint flips")
    p.add_argument("--flip-prob", type=float, default=0.5)
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=1)

    return p.parse_args()


def process(job):
    base, args, errors = job
    pitch, yaw, roll, fov = read_gt(os.path.join(args.in_dir, f"{base}_gt.csv"))
    pano = np.array(Image.open(os.path.join(args.pano_dir, f"{base}_panorama.jpg")).convert("RGB"))
    query = np.array(Image.open(os.path.join(args.in_dir, f"{base}_query.jpg")).convert("RGB"))
    min_x, max_x = content_extent_x(pano)
    width_ratio = (max_x - min_x + 1) / float(pano.shape[1])

    rng = np.random.default_rng((zlib.crc32(base.encode()) + args.seed) % 2 ** 32)
    deltas = sample_offsets(rng, args.per_image, args.crop_deg, fov, args.offset_mode, errors, args.max_delta)
    flips = rng.random(len(deltas)) < args.flip_prob if args.augment else np.zeros(len(deltas), bool)
    seeds = rng.integers(0, 2 ** 31 - 1, size=len(deltas))

    for k, delta in enumerate(deltas):
        center = wrap(yaw + float(delta))
        p_w, y_w, r_w = real_to_window(center, pitch, yaw, roll, args.crop_deg, bool(flips[k]), width_ratio)
        p_w = float(np.clip(p_w, -PITCH_LIMIT, PITCH_LIMIT))
        win = crop_window(pano, center, args.crop_deg, (args.out_height, args.out_width), (min_x, max_x))
        q = query
        if flips[k]:
            q, win = np.ascontiguousarray(q[:, ::-1]), np.ascontiguousarray(win[:, ::-1])
        if args.augment:
            q = color_jitter_keep_black(q, int(seeds[k]))

        out = os.path.join(args.out_dir, f"{base}_o{k}")
        Image.fromarray(win).save(out + "_panorama.jpg", "JPEG", quality=args.quality, subsampling=0)
        Image.fromarray(q).save(out + "_query.jpg", "JPEG", quality=args.quality, subsampling=0)
        with open(out + "_gt.csv", "w", newline="") as f:
            csv.writer(f).writerow([p_w, y_w, r_w, fov])

    return base


def main():
    args = parse_args()
    args.pano_dir = args.pano_dir or args.in_dir
    os.makedirs(args.out_dir, exist_ok=True)
    errors = None
    if args.offset_mode == "empirical":
        with open(args.errors_csv) as f:
            errors = np.array([float(r["yaw_err_signed"]) for r in csv.DictReader(f)])

    bases = sorted(f[:-len("_gt.csv")] for f in os.listdir(args.in_dir) if f.endswith("_gt.csv"))
    jobs = [(b, args, errors) for b in bases]
    if args.workers > 1:
        with Pool(args.workers) as pool:
            for i, b in enumerate(pool.imap_unordered(process, jobs), 1):
                print(f"{i}/{len(jobs)} {b}", flush=True)
    else:
        for i, job in enumerate(jobs, 1):
            print(f"{i}/{len(jobs)} {process(job)}", flush=True)


if __name__ == "__main__":
    main()
