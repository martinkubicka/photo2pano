import numpy as np
import torch


def padding_bands(arr, black_thresh=5, band_frac=0.95):
    """(top, bottom, left, right) extent of the letterbox bands of an RGB uint8 image.

    A row (column) belongs to a band if at least band_frac of its pixels are black; the bands are found
    by scanning inward from each border.
    """
    black = (arr <= black_thresh).all(axis=2)
    H, W = black.shape
    rows, cols = black.mean(axis=1), black.mean(axis=0)

    t = 0
    while t < H and rows[t] >= band_frac:
        t += 1
    b = H
    while b > 0 and rows[b - 1] >= band_frac:
        b -= 1
    l = 0
    while l < W and cols[l] >= band_frac:
        l += 1
    r = W
    while r > 0 and cols[r - 1] >= band_frac:
        r -= 1

    return t, b, l, r


def black_patch_mask(pil_img, patch_size=16, black_thresh=5, black_frac=0.02, band_frac=0.95):
    """Flattened patch mask of the photo (True = padding token).

    A whole patch row (column) is masked as soon as it overlaps a padding band by at least black_frac.
    """
    arr = np.asarray(pil_img)
    H, W = arr.shape[:2]
    p = patch_size
    t, b, l, r = padding_bands(arr, black_thresh, band_frac)

    grid = np.zeros((H // p, W // p), dtype=bool)
    rows = np.arange(H // p) * p
    grid[np.clip(t - rows, 0, p) / p >= black_frac, :] = True
    grid[np.clip(rows + p - b, 0, p) / p >= black_frac, :] = True
    cols = np.arange(W // p) * p
    grid[:, np.clip(l - cols, 0, p) / p >= black_frac] = True
    grid[:, np.clip(cols + p - r, 0, p) / p >= black_frac] = True

    return torch.from_numpy(grid.reshape(-1))
