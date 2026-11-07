"""GeoPose3K -> photo (512x512), panorama canvas (4096x2048) and gt csv (pitch, yaw, roll, hfov in deg).

    python -m preprocessing.geopose3k --dataset geoPose3K_final_publish --panos geoPose3K-panos --out data/gp3k

--panos holds one folder per sample with the cylindrical rendering in cyl/pano.{jpg,png}.
"""
import argparse
import math
import os

from PIL import Image

from .common import save_sample, wrap

Image.MAX_IMAGE_PIXELS = None


def find(folder, stem):
    for ext in (".jpg", ".jpeg", ".png"):
        if os.path.isfile(os.path.join(folder, stem + ext)):
            return os.path.join(folder, stem + ext)

    return None


def read_info(path):
    with open(path) as f:
        lines = [l.strip() for l in f]
    yaw, pitch, roll = map(float, lines[1].split())
    fov = math.degrees(float(lines[5]))
    yaw = wrap(math.degrees(math.pi - (yaw % (-2 * math.pi))))
    pitch = math.degrees(pitch)
    roll = wrap(wrap(math.degrees(-(roll % (-2 * math.pi)))) / 2.0)

    return pitch, yaw, roll, fov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--panos", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for base in sorted(os.listdir(args.panos)):
        pano = find(os.path.join(args.panos, base, "cyl"), "pano")
        photo = find(os.path.join(args.dataset, base), "photo")
        info = os.path.join(args.dataset, base, "info.txt")
        if not (pano and photo and os.path.isfile(info)):
            print(f"incomplete sample {base} - skipping")
            continue
        save_sample(os.path.join(args.out, base), Image.open(photo), Image.open(pano), *read_info(info))


if __name__ == "__main__":
    main()
