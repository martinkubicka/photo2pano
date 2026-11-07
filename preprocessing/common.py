import csv

from PIL import Image, ImageOps

QUERY_SIZE = (512, 512)
PANO_SIZE = (4096, 2048)


def wrap(angle):
    return (angle + 180.0) % 360.0 - 180.0


def letterbox(img, size, resample=Image.LANCZOS):
    """Resize with preserved aspect ratio and zero padding to `size` (w, h), centered."""
    return ImageOps.pad(img, size, method=resample, color=(0, 0, 0))


def save_sample(out_base, query, pano, pitch, yaw, roll, fov, resample=Image.LANCZOS):
    letterbox(query.convert("RGB"), QUERY_SIZE, resample).save(out_base + "_query.jpg", quality=100, subsampling=0)
    letterbox(pano.convert("RGB"), PANO_SIZE, resample).save(out_base + "_panorama.jpg", quality=100, subsampling=0)
    with open(out_base + "_gt.csv", "w", newline="") as f:
        csv.writer(f).writerow([f"{pitch:.6f}", f"{yaw:.6f}", f"{roll:.6f}", f"{fov:.6f}"])
