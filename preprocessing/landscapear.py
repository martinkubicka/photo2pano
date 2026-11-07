"""LandscapeAR -> photo, panorama canvas and gt csv (pitch, yaw, roll, hfov in deg).

    python -m preprocessing.landscapear --photos query_photos --panos new_pano_dataset \
        --info new_pano_dataset/datasetInfoClean.csv --out data/lar [--exclude removed_pairs.txt]

--panos holds folders lsar_<name>/cyl/pano.png; --info lists name, ..., yaw, pitch, roll, fov (radians)
in its last four columns; --exclude lists the names of the removed pairs.
"""
import argparse
import csv
import math
import os

from PIL import Image

from .common import save_sample, wrap

Image.MAX_IMAGE_PIXELS = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", required=True)
    ap.add_argument("--panos", required=True)
    ap.add_argument("--info", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    excluded = set()
    if args.exclude:
        with open(args.exclude) as f:
            excluded = {l.strip() for l in f if l.strip()}

    with open(args.info) as f:
        for row in csv.reader(f):
            name = row[1].strip()
            if name in excluded:
                continue
            photo = os.path.join(args.photos, name + ".jpg")
            pano = os.path.join(args.panos, "lsar_" + name, "cyl", "pano.png")
            if not (os.path.isfile(photo) and os.path.isfile(pano)):
                continue
            yaw, pitch, roll, fov = (math.degrees(float(v)) for v in row[-4:])
            save_sample(os.path.join(args.out, name), Image.open(photo), Image.open(pano),
                        pitch, wrap(180.0 - yaw), -roll, fov)


if __name__ == "__main__":
    main()
