import os

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from .padding import black_patch_mask

TRANSFORM = T.Compose([T.ToTensor(), T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])


def read_gt(path):
    """pitch, yaw, roll, horizontal FoV in degrees (one csv line)."""
    with open(path) as f:
        pitch, yaw, roll, fov = map(float, f.readline().strip().split(",")[:4])

    return pitch, yaw, roll, fov


def list_samples(root, split=None):
    """Sample base names in root ({base}_query.jpg, {base}_panorama.jpg, {base}_gt.csv), optionally
    restricted to a split file with one base name per line."""
    bases = sorted(f[:-len("_gt.csv")] for f in os.listdir(root) if f.endswith("_gt.csv"))
    bases = [b for b in bases if os.path.isfile(os.path.join(root, f"{b}_query.jpg"))
             and os.path.isfile(os.path.join(root, f"{b}_panorama.jpg"))]
    if split:
        with open(split) as f:
            keep = {os.path.basename(l.strip().split(",")[0]).replace("_panorama.jpg", "") for l in f if l.strip()}
        bases = [b for b in bases if b in keep]

    return bases


class OrientationDataset(Dataset):
    """(photo tensor, panorama path, photo padding mask, pitch, yaw, roll, fov)."""

    def __init__(self, root, bases=None, transform=TRANSFORM):
        self.root = root
        self.transform = transform
        self.bases = bases if bases is not None else list_samples(root)
        if not self.bases:
            raise RuntimeError(f"no samples found in {root}")

    def __len__(self):
        return len(self.bases)

    def __getitem__(self, idx):
        b = self.bases[idx]
        query = Image.open(os.path.join(self.root, f"{b}_query.jpg")).convert("RGB")
        pitch, yaw, roll, fov = read_gt(os.path.join(self.root, f"{b}_gt.csv"))

        return (self.transform(query), os.path.join(self.root, f"{b}_panorama.jpg"), black_patch_mask(query),
                pitch, yaw, roll, fov)
