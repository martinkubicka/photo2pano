"""Cross-attention maps and padding statistics of a Photo2Pano checkpoint.

    python -m tools.attention --data geopose_test --ckpt stage1.pth --samples b1 b2 --out maps.npz
    python -m tools.attention --data geopose_test --ckpt stage1.pth --random 40 --no-mask --out stats.npz

Per sample it stores the panorama -> photo attention averaged over panorama tokens and heads (32x32
photo grid), the photo -> panorama attention of the non-padding photo tokens per tile (32 x 32x32), the
attention mass and the pooling weight on the padding tokens, and the prediction.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from photo2pano import OrientationEstimator
from photo2pano.angles import angle_centers, angle_errors, circ_expect_deg
from photo2pano.data import TRANSFORM, list_samples, read_gt
from photo2pano.padding import black_patch_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--samples", nargs="*", default=[])
    ap.add_argument("--random", type=int, default=0)
    ap.add_argument("--no-mask", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OrientationEstimator(device=device).to(device)
    state = torch.load(args.ckpt, map_location=device)["model_state"]
    model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
    model.eval()
    cp, cy, cr = angle_centers(device)

    bases = list(args.samples)
    if args.random:
        allb = list_samples(args.data)
        bases += [allb[i] for i in sorted(np.random.default_rng(0).choice(len(allb), args.random, replace=False))]

    out, stats = {}, []
    for b in bases:
        q_pil = Image.open(os.path.join(args.data, f"{b}_query.jpg")).convert("RGB")
        pad = black_patch_mask(q_pil)
        mask = None if args.no_mask else pad.unsqueeze(0).to(device)
        with torch.no_grad():
            lp, ly, lr, att = model(TRANSFORM(q_pil).unsqueeze(0).to(device),
                                    [os.path.join(args.data, f"{b}_panorama.jpg")], mask, return_attention=True)
        pred = tuple(circ_expect_deg(F.softmax(l, 1), c)[0].item() for l, c in ((lp, cp), (ly, cy), (lr, cr)))
        err = angle_errors(read_gt(os.path.join(args.data, f"{b}_gt.csv"))[:3], pred)["rot_err"]

        pad = pad.numpy()
        a_pq = att["pano_to_photo"][0].cpu().numpy()
        a_qp = att["photo_to_pano"][0].cpu().numpy()
        w_q = att["pool_photo"][0].cpu().numpy()
        keep = ~pad if pad.any() else np.ones_like(pad)
        g = int(np.sqrt(att["Np"]))
        out[f"{b}|photo"] = a_pq.mean(0).reshape(g, g)
        out[f"{b}|pano_tiles"] = a_qp[keep].mean(0).reshape(-1, g, g)
        out[f"{b}|pred"] = np.array(pred)
        stats.append((pad.mean(), a_pq[:, pad].sum(1).mean(), w_q[pad].sum(), err))
        print(f"{b}: rot {err:.2f}, padding {pad.mean():.3f}, attention on padding {stats[-1][1]:.3f}, "
              f"pooling on padding {stats[-1][2]:.3f}", flush=True)

    s = np.array(stats)
    out["stats"] = s
    np.savez_compressed(args.out, **out)
    print(f"mean over {len(s)}: padding share {s[:, 0].mean():.3f}, attention on padding {s[:, 1].mean():.3f}, "
          f"pooling on padding {s[:, 2].mean():.3f}")


if __name__ == "__main__":
    main()
