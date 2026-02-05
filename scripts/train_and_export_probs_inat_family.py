#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models


# ============================================================
# Dataset
# ============================================================

class NPZImageDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, train: bool, seed: int = 1):
        assert X.ndim == 4 and X.shape[1] == 3, f"Expected (N,3,H,W), got {X.shape}"
        self.X = X
        self.y = y.astype(np.int64)
        self.train = train
        self.rng = np.random.default_rng(seed)

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return self.X.shape[0]

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        _, H, W = x.shape

        # Random resized crop
        scale = float(self.rng.uniform(0.6, 1.0))
        new_h = int(H * scale)
        new_w = int(W * scale)

        top = int(self.rng.integers(0, H - new_h + 1))
        left = int(self.rng.integers(0, W - new_w + 1))

        x = x[:, top:top + new_h, left:left + new_w]
        x = F.interpolate(x.unsqueeze(0), size=(H, W),
                          mode="bilinear", align_corners=False).squeeze(0)

        if self.rng.random() < 0.5:
            x = torch.flip(x, dims=[2])

        return x

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float() / 255.0
        if self.train:
            x = self._augment(x)
        x = (x - self.mean) / self.std
        y = int(self.y[idx])
        return x, y


# ============================================================
# Model
# ============================================================

def build_resnet50(num_classes: int, finetune: str = "full"):
    finetune = finetune.lower()
    assert finetune in ["full", "head", "last"]

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    def set_grad(m, flag):
        for p in m.parameters():
            p.requires_grad = flag

    if finetune == "head":
        set_grad(model, False)
        set_grad(model.fc, True)
    elif finetune == "last":
        set_grad(model, False)
        set_grad(model.layer4, True)
        set_grad(model.fc, True)

    return model


# ============================================================
# Train / Eval / Predict
# ============================================================

def train_one_epoch(model, loader, opt, scaler, device, amp: bool):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        loss_sum += float(loss.detach()) * x.size(0)
        correct += int((logits.argmax(1) == y).sum())
        total += int(x.size(0))

    return loss_sum / total, correct / total


@torch.no_grad()
def eval_acc(model, loader, device, amp: bool):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y)

        loss_sum += float(loss.detach()) * x.size(0)
        correct += int((logits.argmax(1) == y).sum())
        total += int(x.size(0))

    return loss_sum / total, correct / total


@torch.no_grad()
def predict_proba(model, loader, device, amp: bool):
    model.eval()
    Ps, Zs, Ys = [], [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
            p = torch.softmax(logits, dim=1)

        Ps.append(p.cpu().numpy())
        Zs.append(logits.cpu().numpy())
        Ys.append(y.numpy())

    return (
        np.concatenate(Ps).astype(np.float32),
        np.concatenate(Zs).astype(np.float32),
        np.concatenate(Ys).astype(np.int64),
    )


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_npz", type=str, required=True)
    ap.add_argument("--out_npz", type=str, required=True)

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--finetune", type=str, default="full", choices=["full", "head", "last"])

    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--tail_frac", type=float, default=0.2)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = args.amp and (device.type == "cuda")

    z = np.load(args.img_npz, allow_pickle=True)
    X_train, y_train = z["X_train"], z["y_train"]
    X_sel,   y_sel   = z["X_sel"],   z["y_sel"]
    X_cal,   y_cal   = z["X_cal"],   z["y_cal"]
    X_test,  y_test  = z["X_test"],  z["y_test"]

    K = int(max(y_train.max(), y_sel.max(), y_cal.max(), y_test.max())) + 1

    dl_args = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    dl_train = DataLoader(NPZImageDataset(X_train, y_train, True, args.seed), shuffle=True, **dl_args)
    dl_sel   = DataLoader(NPZImageDataset(X_sel,   y_sel,   False), shuffle=False, **dl_args)
    dl_cal   = DataLoader(NPZImageDataset(X_cal,   y_cal,   False), shuffle=False, **dl_args)
    dl_test  = DataLoader(NPZImageDataset(X_test,  y_test,  False), shuffle=False, **dl_args)

    model = build_resnet50(K, finetune=args.finetune).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    best_acc = -1.0
    best_state = None

    print(f"[train] epochs={args.epochs}, lr={args.lr}, finetune={args.finetune}")
    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, dl_train, opt, scaler, device, amp)
        va_loss, va_acc = eval_acc(model, dl_sel, device, amp)
        print(f"[ep {ep:02d}] train acc {tr_acc:.4f} | sel acc {va_acc:.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    # Extract probabilities
    p_sel,  z_sel,  _ = predict_proba(model, dl_sel,  device, amp)
    p_cal,  z_cal,  _ = predict_proba(model, dl_cal,  device, amp)
    p_test, z_test, _ = predict_proba(model, dl_test, device, amp)

    counts_pool = np.bincount(y_train.astype(int), minlength=K)
    m = int(np.ceil(args.tail_frac * K))
    tail_set = np.argsort(counts_pool)[:m]

    meta = dict(
        img_npz=args.img_npz,
        model="resnet50",
        epochs=args.epochs,
        lr=args.lr,
        finetune=args.finetune,
        best_sel_acc=float(best_acc),
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        p_sel=p_sel, y_sel=y_sel,
        p_cal=p_cal, y_cal=y_cal,
        p_test=p_test, y_test=y_test,
        z_sel=z_sel, z_cal=z_cal, z_test=z_test,
        counts_pool=counts_pool,
        tail_set=tail_set,
        meta=np.array([meta], dtype=object),
    )

    print(f"[saved] {args.out_npz}")
    print("[shapes]",
          "p_sel", p_sel.shape,
          "p_cal", p_cal.shape,
          "p_test", p_test.shape)


if __name__ == "__main__":
    main()
