"""Simulated first-stage yaw errors for training the second stage.

Training windows are centered at yaw_gt + delta. "mixture" draws |delta| from 55% |N(0, 6)|,
30% U(6, 45) and 15% U(45, 75) with a random sign (E|delta| ~ 19 deg); "empirical" resamples measured
signed first-stage errors (sign-symmetrized, 1 deg jitter). Offsets are clipped so that the photo stays
inside the window: |delta| <= min(max_delta, (crop_deg - fov) / 2).
"""
import numpy as np

MIXTURE = [(0.55, "halfnormal", 6.0, 0.0), (0.30, "uniform", 6.0, 45.0), (0.15, "uniform", 45.0, 75.0)]


def sample_mixture(rng, n, mixture=MIXTURE):
    w = np.array([m[0] for m in mixture])
    comp = rng.choice(len(mixture), size=n, p=w / w.sum())
    out = np.zeros(n)
    for i, (_, kind, a, b) in enumerate(mixture):
        sel = comp == i
        k = int(sel.sum())
        if kind == "halfnormal":
            out[sel] = np.abs(rng.normal(0.0, a, size=k))
        else:
            out[sel] = rng.uniform(a, b, size=k)

    return out * rng.choice([-1.0, 1.0], size=n)


def sample_empirical(rng, n, errors, jitter=1.0):
    errors = np.asarray(errors, dtype=np.float64)
    out = rng.choice(np.concatenate([errors, -errors]), size=n, replace=True)

    return out + rng.normal(0.0, jitter, size=n) if jitter > 0 else out


def max_delta(crop_deg, fov_deg, hard_cap=75.0):
    return max(0.0, min(hard_cap, (float(crop_deg) - float(fov_deg)) / 2.0))


def sample_offsets(rng, n, crop_deg, fov_deg, mode="mixture", errors=None, hard_cap=75.0):
    if mode == "mixture":
        d = sample_mixture(rng, n)
    elif mode == "empirical":
        d = sample_empirical(rng, n, errors)
    else:
        raise ValueError(f"unknown offset mode {mode}")
    cap = max_delta(crop_deg, fov_deg, hard_cap)

    return np.clip(d, -cap, cap)
