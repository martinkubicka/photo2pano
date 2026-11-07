import torch
import torch.nn.functional as F

from .angles import circ_expect_deg, lin_expect_deg


def huber(d, delta):
    return torch.where(d <= delta, 0.5 * d ** 2 / (delta + 1e-12), d - 0.5 * delta)


def circular_huber_deg(pred_deg, tgt_deg, delta_deg=3.0, period_deg=360.0):
    """Huber loss (smooth-L1 form, in radians) on the wrapped angular difference."""
    pred, tgt = torch.deg2rad(pred_deg), torch.deg2rad(tgt_deg)
    period = torch.deg2rad(torch.tensor(period_deg, device=pred.device, dtype=pred.dtype))
    d = torch.abs((pred - tgt + 0.5 * period) % period - 0.5 * period)
    delta = torch.deg2rad(torch.tensor(delta_deg, device=pred.device, dtype=pred.dtype))

    return huber(d, delta)


def orientation_loss(logits, targets, centers, delta_deg=3.0):
    """Mean of the circular Huber losses of pitch, yaw and roll on the circular soft-argmax."""
    periods = (180.0, 360.0, 360.0)
    losses, preds = [], []
    for lg, tgt, c, per in zip(logits, targets, centers, periods):
        pred = circ_expect_deg(F.softmax(lg.float(), dim=1), c)
        losses.append(circular_huber_deg(pred, tgt, delta_deg, per).mean())
        preds.append(pred)

    return sum(losses) / 3.0, preds


def fov_loss(logits_fov, tgt_deg, centers, delta_deg=3.0, w_ce=0.1, sigma_deg=2.0):
    """Huber on the linear soft-argmax plus a soft-label term towards N(tgt, sigma).

    The soft-label term is written as a KL divergence: same gradients as the cross-entropy, but it is
    zero for a perfect prediction, which keeps the logged loss readable.
    """
    prob = F.softmax(logits_fov.float(), dim=1)
    pred = lin_expect_deg(prob, centers)
    hub = huber(torch.abs(torch.deg2rad(pred) - torch.deg2rad(tgt_deg)),
                torch.deg2rad(torch.tensor(delta_deg, device=pred.device))).mean()

    c = centers.to(tgt_deg.device).unsqueeze(0)
    soft = torch.exp(-0.5 * ((tgt_deg.unsqueeze(1) - c) / sigma_deg) ** 2)
    soft = soft / soft.sum(1, keepdim=True).clamp_min(1e-12)
    ce = -(soft * F.log_softmax(logits_fov.float(), dim=1)).sum(1)
    ent = -(soft * torch.log(soft.clamp_min(1e-12))).sum(1)

    return hub + w_ce * (ce - ent).mean(), pred
