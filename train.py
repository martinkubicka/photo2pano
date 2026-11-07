"""Train the first stage or the second stage of Photo2Pano.

Both stages use the same loop: a dataset directory of ({base}_query.jpg, {base}_panorama.jpg,
{base}_gt.csv). For the second stage, the directory is built by build_stage2_dataset.py and holds
windows with labels in the window frame.

    torchrun --standalone --nproc_per_node=1 train.py --data DATA --out checkpoints/stage1
    torchrun --standalone --nproc_per_node=1 train.py --data DATA_S2 --out checkpoints/stage2 --epochs 50
    torchrun ... train.py --data DATA --out checkpoints/stage1_fov --fov-head
    torchrun ... train.py --data DATA_S2 --out ... --init checkpoints/stage1/model_25.pth --lr 3e-5 --lr-backbone 3e-6
"""
import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from photo2pano import OrientationEstimator
from photo2pano.angles import angle_centers, class_centers_deg, rotation_error_deg, wrap
from photo2pano.data import OrientationDataset, list_samples
from photo2pano.losses import fov_loss, orientation_loss

W_FOV = 0.25


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default=None, help="optional file with the training base names")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--fov-head", action="store_true")
    p.add_argument("--init", default=None, help="initialize from a checkpoint (fine-tuning)")
    p.add_argument("--resume", default=None, help="resume model and optimizer state")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(model, loader, device, optimizer=None, scaler=None, fov_head=False, is_main=True):
    train = optimizer is not None
    model.train(train)
    centers = angle_centers(device)
    fov_centers = class_centers_deg(180, 0.0).to(device)
    totals = torch.zeros(4, device=device)   # loss, rotation error, fov error, samples

    for query, pano, mask, pitch, yaw, roll, fov in tqdm(loader, disable=not is_main,
                                                         desc="train" if train else "val"):
        query, mask = query.to(device, non_blocking=True), mask.to(device, non_blocking=True)
        tgt = [wrap(pitch.float(), 180.0).to(device), wrap(yaw.float()).to(device), wrap(roll.float()).to(device)]

        with torch.set_grad_enabled(train), autocast():
            out = model(query, pano, mask)
            loss, preds = orientation_loss(out[:3], tgt, centers)
            fov_err = torch.zeros(1, device=device)
            if fov_head:
                tgt_fov = fov.float().to(device).clamp(0.0, 180.0 - 1e-3)
                l_fov, pred_fov = fov_loss(out[3], tgt_fov, fov_centers)
                loss = loss + W_FOV * l_fov
                fov_err = (pred_fov.detach() - tgt_fov).abs()

        if train:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            p, y, r = (x.detach().float().cpu().numpy() for x in preds)
            rot = [rotation_error_deg((pitch[j].item(), yaw[j].item(), roll[j].item()), (p[j], y[j], r[j]))
                   for j in range(len(p))]
            bs = query.size(0)
            totals += torch.tensor([loss.item() * bs, float(np.sum(rot)), float(fov_err.sum()), bs], device=device)

    if dist.is_initialized():
        dist.all_reduce(totals)
    n = totals[3].item()

    return totals[0].item() / n, totals[1].item() / n, totals[2].item() / n


def main():
    args = parse_args()
    dist.init_process_group(backend="nccl")
    local_rank, rank = int(os.environ["LOCAL_RANK"]), int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_main = rank == 0
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    ds = OrientationDataset(args.data, list_samples(args.data, args.split))
    # the split is drawn over the augmented samples, so the validation set shares photos with the
    # training set; it is only used to monitor training
    idx_train, idx_val = train_test_split(range(len(ds)), test_size=args.val_frac, random_state=args.seed,
                                          shuffle=True)
    train_set, val_set = Subset(ds, idx_train), Subset(ds, idx_val)
    train_sampler = DistributedSampler(train_set, shuffle=True, seed=args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    dl_train = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler, pin_memory=True,
                          generator=g)
    dl_val = DataLoader(val_set, batch_size=args.batch_size, sampler=DistributedSampler(val_set, shuffle=False),
                        pin_memory=True)

    model = OrientationEstimator(device=device, fov_head=args.fov_head).to(device)
    if args.init:
        state = torch.load(args.init, map_location=device)["model_state"]
        missing, unexpected = model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()},
                                                    strict=False)
        if is_main:
            print(f"initialized from {args.init} (missing {missing}, unexpected {unexpected})")
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    backbone = list(model.module.backbone.parameters())
    rest = [p for n, p in model.named_parameters() if "backbone." not in n]
    opt = torch.optim.AdamW([{"params": backbone, "lr": args.lr_backbone}, {"params": rest, "lr": args.lr}])
    scaler = GradScaler()
    start = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.module.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["optim_state"])
        start = ckpt.get("epoch", 0) + 1

    log = open(os.path.join(args.out, "training_log.txt"), "a") if is_main else None
    for ep in range(start, args.epochs + 1):
        train_sampler.set_epoch(ep)
        tr = run_epoch(model, dl_train, device, opt, scaler, args.fov_head, is_main)
        torch.cuda.empty_cache()
        va = run_epoch(model, dl_val, device, fov_head=args.fov_head, is_main=is_main)
        if is_main:
            torch.save({"model_state": model.module.state_dict(), "optim_state": opt.state_dict(), "epoch": ep},
                       os.path.join(args.out, f"model_{ep}.pth"))
            msg = (f"epoch {ep} | train loss {tr[0]:.3f}, rot {tr[1]:.2f}, fov {tr[2]:.2f} | "
                   f"val loss {va[0]:.3f}, rot {va[1]:.2f}, fov {va[2]:.2f}")
            print(msg)
            log.write(msg + "\n")
            log.flush()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
