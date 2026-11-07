"""Evaluate Photo2Pano (one or two stages) on a dataset directory.

    python evaluate.py --data geopose_test --stage1 stage1.pth --out results/geopose.csv
    python evaluate.py --data geopose_test --stage1 stage1.pth --stage2 stage2.pth --out results/geopose_2s.csv
    python evaluate.py ... --stage1-fov-head          # stage 1 with FoV head

Writes one row per image (predictions, confidences, errors) and prints mean / median errors, accuracies
and the AUC of the cumulative error curve on [0, 180] deg over all images (for Venturi: all frames, as in
the paper). For Venturi (base names venturi_<seq>_*), the mean error of each sequence is printed as well.
"""
import argparse
import csv
import io
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from photo2pano import OrientationEstimator
from photo2pano.angles import angle_centers, angle_errors, circ_expect_deg, class_centers_deg, lin_expect_deg, \
    resultant_length
from photo2pano.data import TRANSFORM, list_samples, read_gt
from photo2pano.padding import black_patch_mask
from photo2pano.window import content_extent_x, crop_window, window_to_real

Image.MAX_IMAGE_PIXELS = None
THRESHOLDS = (1, 2, 5, 10, 20, 45)


def load(path, device, fov_head=False):
    model = OrientationEstimator(device=device, fov_head=fov_head).to(device)
    state = torch.load(path, map_location=device)["model_state"]
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})

    return model.eval()


@torch.no_grad()
def predict(model, query, pano, mask, centers):
    out = model(query, [pano], mask)
    cp, cy, cr = centers
    prob_y = F.softmax(out[1], 1)
    angles = (circ_expect_deg(F.softmax(out[0], 1), cp)[0].item(), circ_expect_deg(prob_y, cy)[0].item(),
              circ_expect_deg(F.softmax(out[2], 1), cr)[0].item())
    conf = resultant_length(prob_y, cy)[0].item()
    fov = lin_expect_deg(F.softmax(out[3].float(), 1), class_centers_deg(180, 0.0))[0].item() if len(out) > 3 \
        else float("nan")

    return angles, conf, fov


def summary(err):
    err = np.asarray(err)
    s = {"mean": err.mean(), "median": np.median(err), "auc": 1.0 - np.minimum(err, 180).mean() / 180.0}
    s.update({f"<{t}": (err < t).mean() * 100 for t in THRESHOLDS})

    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", default=None)
    ap.add_argument("--stage1-fov-head", action="store_true")
    ap.add_argument("--pano2-dir", default=None, help="panoramas for the window (default: --data)")
    ap.add_argument("--crop-deg", type=float, default=180.0)
    ap.add_argument("--no-mask", action="store_true", help="evaluate without the padding mask")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    centers = angle_centers(device)
    m1 = load(args.stage1, device, args.stage1_fov_head)
    m2 = load(args.stage2, device) if args.stage2 else None
    bases = list_samples(args.data)[: args.limit or None]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows = []
    for i, b in enumerate(bases):
        query_pil = Image.open(os.path.join(args.data, f"{b}_query.jpg")).convert("RGB")
        query = TRANSFORM(query_pil).unsqueeze(0).to(device)
        mask = None if args.no_mask else black_patch_mask(query_pil).unsqueeze(0).to(device)
        gt = read_gt(os.path.join(args.data, f"{b}_gt.csv"))
        pano = os.path.join(args.data, f"{b}_panorama.jpg")

        a1, conf1, fov1 = predict(m1, query, pano, mask, centers)
        row = {"base": b, "pitch_gt": gt[0], "yaw_gt": gt[1], "roll_gt": gt[2], "fov_gt": gt[3],
               "pitch_s1": a1[0], "yaw_s1": a1[1], "roll_s1": a1[2], "conf_s1": conf1, "fov_s1": fov1}
        row.update({f"s1_{k}": v for k, v in angle_errors(gt[:3], a1).items()})

        if m2 is not None:
            pano_np = np.array(Image.open(os.path.join(args.pano2_dir or args.data, f"{b}_panorama.jpg"))
                               .convert("RGB"))
            min_x, max_x = content_extent_x(pano_np)
            width_ratio = (max_x - min_x + 1) / float(pano_np.shape[1])
            win = crop_window(pano_np, a1[1], args.crop_deg, content_x=(min_x, max_x))
            buf = io.BytesIO()
            Image.fromarray(win).save(buf, "JPEG", quality=100, subsampling=0)
            buf.seek(0)
            aw, conf2, _ = predict(m2, query, buf, mask, centers)
            a2 = window_to_real(a1[1], aw[0], aw[1], aw[2], args.crop_deg, width_ratio)
            row.update({"pitch_s2": a2[0], "yaw_s2": a2[1], "roll_s2": a2[2], "conf_s2": conf2})
            row.update({f"s2_{k}": v for k, v in angle_errors(gt[:3], a2).items()})
        rows.append(row)
        print(f"{i + 1}/{len(bases)} {b}: s1 {row['s1_rot_err']:.2f}"
              + (f", s2 {row['s2_rot_err']:.2f}" if m2 is not None else ""), flush=True)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for stage in ("s1", "s2") if m2 is not None else ("s1",):
        err = [r[f"{stage}_rot_err"] for r in rows]
        s = summary(err)
        print(f"{stage}: mean {s['mean']:.2f}, median {s['median']:.2f}, AUC {s['auc']:.3f}, "
              + ", ".join(f"<{t}: {s[f'<{t}']:.1f}%" for t in THRESHOLDS))
        seq = defaultdict(list)
        for r in rows:
            if r["base"].startswith("venturi_"):
                seq[r["base"].split("_")[1]].append(r[f"{stage}_rot_err"])
        if seq:
            print(f"{stage}: per-sequence means " + ", ".join(f"{k.upper()} {np.mean(v):.2f}" for k, v in sorted(seq.items())))
    if args.stage1_fov_head:
        fe = np.abs([r["fov_s1"] - r["fov_gt"] for r in rows])
        print(f"FoV: mean {fe.mean():.2f}, median {np.median(fe):.2f}")


if __name__ == "__main__":
    main()
