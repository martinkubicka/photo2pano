import numpy as np
import torch
from scipy.spatial.transform import Rotation

PITCH_BINS, YAW_BINS, ROLL_BINS = 180, 360, 360


def wrap(angle, period=360.0):
    return (angle + period / 2.0) % period - period / 2.0


def class_centers_deg(n_bins, start_deg):
    return start_deg + (torch.arange(n_bins, dtype=torch.float32) + 0.5)


def angle_centers(device="cpu"):
    """Bin centers of the pitch, yaw and roll heads (1 deg bins)."""
    return (class_centers_deg(PITCH_BINS, -90.0).to(device),
            class_centers_deg(YAW_BINS, -180.0).to(device),
            class_centers_deg(ROLL_BINS, -180.0).to(device))


def circ_expect_deg(prob, centers_deg):
    """Circular soft-argmax: circular mean of the bin centers weighted by prob."""
    theta = torch.deg2rad(centers_deg).to(prob.device)

    return torch.rad2deg(torch.atan2(prob @ torch.sin(theta), prob @ torch.cos(theta)))


def resultant_length(prob, centers_deg):
    """Mean resultant length of a circular distribution, used as confidence (1 = sharp, 0 = uniform)."""
    theta = torch.deg2rad(centers_deg).to(prob.device)

    return torch.sqrt((prob @ torch.cos(theta)) ** 2 + (prob @ torch.sin(theta)) ** 2)


def lin_expect_deg(prob, centers_deg):
    return prob @ centers_deg.to(prob.device)


def rotation_error_deg(gt, pred):
    """Geodesic distance between two orientations given as (pitch, yaw, roll) in degrees."""
    R_gt = Rotation.from_euler("zyx", [gt[1], gt[0], gt[2]], degrees=True).as_matrix()
    R_pr = Rotation.from_euler("zyx", [pred[1], pred[0], pred[2]], degrees=True).as_matrix()
    c = np.clip((np.trace(R_gt.T @ R_pr) - 1.0) / 2.0, -1.0, 1.0)

    return float(np.degrees(np.arccos(c)))


def angle_errors(gt, pred):
    return {
        "rot_err": rotation_error_deg(gt, pred),
        "pitch_err": abs(wrap(pred[0] - gt[0], 180.0)),
        "yaw_err": abs(wrap(pred[1] - gt[1])),
        "roll_err": abs(wrap(pred[2] - gt[2])),
    }
