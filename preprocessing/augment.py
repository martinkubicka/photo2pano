"""Offline augmentation of the first-stage training set.

Every sample is expanded into N_AUGS copies with colour jitter of the photo (the letterbox padding is
kept black) and, with probability 0.5, a joint horizontal flip of photo and panorama, which negates
yaw and roll.

    python -m preprocessing.augment --in-dir data/train_raw --out-dir data/train
"""
import argparse
import csv
import os
import random

import numpy as np
from PIL import Image

BLACK_THRESHOLD = 5
N_AUGS = 6

# (probability, transform); the Bernoulli draws are made here and every transform gets its own seed,
# so that the transforms fire independently and a rebuild reproduces the dataset
TRANSFORMS = [
    (0.6, lambda A: A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0)),
    (0.5, lambda A: A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=15, p=1.0)),
    (0.4, lambda A: A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0)),
    (0.3, lambda A: A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=1.0)),
    (0.4, lambda A: A.RandomGamma(gamma_limit=(80, 120), p=1.0)),
]


def color_jitter_keep_black(img_rgb, seed):
    import albumentations as A

    rng = np.random.default_rng(int(seed))
    black = (img_rgb < BLACK_THRESHOLD).all(axis=2)
    out = img_rgb
    for p, factory in TRANSFORMS:
        if rng.random() >= p:
            continue
        t = factory(A)
        sub = int(rng.integers(0, 2 ** 31 - 1))
        if hasattr(t, "set_random_seed"):
            t.set_random_seed(sub)
        else:
            random.seed(sub)
            np.random.seed(sub)
        out = t(image=out)["image"]
    out = np.array(out, copy=True)
    out[black] = 0

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--copies", type=int, default=N_AUGS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for f in sorted(os.listdir(args.in_dir)):
        if not f.endswith("_query.jpg"):
            continue
        base = f[:-len("_query.jpg")]
        query = np.array(Image.open(os.path.join(args.in_dir, f)).convert("RGB"))
        pano = Image.open(os.path.join(args.in_dir, f"{base}_panorama.jpg")).convert("RGB")
        with open(os.path.join(args.in_dir, f"{base}_gt.csv")) as fh:
            row = next(csv.reader(fh))

        for i in range(args.copies):
            q = color_jitter_keep_black(query, int(rng.integers(0, 2 ** 31 - 1)))
            p = pano
            r = list(row)
            if rng.random() < 0.5:
                q = np.ascontiguousarray(q[:, ::-1])
                p = pano.transpose(Image.FLIP_LEFT_RIGHT)
                r[1], r[2] = f"{-float(r[1]):.6f}", f"{-float(r[2]):.6f}"
            out = os.path.join(args.out_dir, f"{base}_{i}")
            Image.fromarray(q).save(out + "_query.jpg", quality=100, subsampling=0)
            p.save(out + "_panorama.jpg", quality=100, subsampling=0)
            with open(out + "_gt.csv", "w", newline="") as fh:
                csv.writer(fh).writerow(r)


if __name__ == "__main__":
    main()
