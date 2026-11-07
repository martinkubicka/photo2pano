"""Venturi -> photo, panorama canvas and gt csv (pitch, yaw, roll, hfov in deg).

    python -m preprocessing.venturi --venturi venturi --panos venturi_panos --out data/venturi

--venturi holds one folder per frame (venturi_<seq>_<frame>) with image.jpg, rotationC2G.xml and
data.xml; --panos holds one rendering per sequence, venturi_<seq>_00000001/pano.png, rendered at the
position of the first frame.
"""
import argparse
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from .common import save_sample, wrap

Image.MAX_IMAGE_PIXELS = None


def camera_angles(xml_path):
    root = ET.parse(xml_path).getroot()
    right, up, forward = (np.array([float(root.find(c).get(f"x{i}")) for i in range(3)]) for c in ("C0", "C1", "C2"))
    yaw = wrap(np.degrees(np.arctan2(forward[0], forward[1])) + 90.0)
    pitch = -np.degrees(np.arcsin(forward[2]))
    roll = -wrap(np.degrees(np.arctan2(right[2], up[2])))

    return float(pitch), float(yaw), float(roll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venturi", required=True)
    ap.add_argument("--panos", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for folder in sorted(os.listdir(args.venturi)):
        d = os.path.join(args.venturi, folder)
        if not os.path.isdir(d):
            continue
        seq = folder.split("_")[1]
        pano = os.path.join(args.panos, f"venturi_{seq}_00000001", "pano.png")
        files = [os.path.join(d, n) for n in ("image.jpg", "rotationC2G.xml", "data.xml")]
        if not all(os.path.isfile(p) for p in [pano] + files):
            continue
        fov = math.degrees(float(ET.parse(files[2]).getroot().find("field-of-view").get("value")))
        save_sample(os.path.join(args.out, folder), Image.open(files[0]), Image.open(pano),
                    *camera_angles(files[1]), fov, resample=Image.BILINEAR)


if __name__ == "__main__":
    main()
