"""Models: Random Forest, LightGBM (aggregated features) and TempCNN / L-TAE-lite (masked sequences).

Class imbalance (roads are rare):
  * tree models use `class_weight="balanced"` (inverse class frequency);
  * neural models use focal loss (gamma=2) with alpha = 1/sqrt(frequency), normalized to mean 1.
Training samples are also class-capped (see validation.sample_pixels).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(max(1, torch.get_num_threads()))

TABULAR = ("rf", "lgbm")
SEQUENCE = ("tempcnn", "ltae")


# --------------------------------------------------------------------------------------
# tree models
# --------------------------------------------------------------------------------------
def make_tabular(name: str, cfg: dict, seed: int):
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(n_estimators=cfg["run"]["n_arvores"], min_samples_leaf=2, max_features="sqrt", max_samples=0.5, max_depth=25,
                                      class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    if name == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(n_estimators=cfg["run"]["n_arvores"] * 2, learning_rate=0.05,
                              num_leaves=cfg["modelos"]["lgbm_folhas"], subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, class_weight="balanced", random_state=seed, n_jobs=-1,
                              verbose=-1)
    raise ValueError(name)


def tabular_proba(model, F_: np.ndarray, n_classes: int, chunk: int = 200_000) -> np.ndarray:
    out = np.zeros((len(F_), n_classes), np.float32)
    cls = model.classes_.astype(int)
    for i in range(0, len(F_), chunk):
        out[i:i + chunk, cls] = model.predict_proba(F_[i:i + chunk])
    return out


# --------------------------------------------------------------------------------------
# neural models
# --------------------------------------------------------------------------------------
class TempCNN(nn.Module):
    """Pelletier et al. (2019) TempCNN; global pooling instead of flatten so T may vary."""

    def __init__(self, n_in: int, n_classes: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        layers, c = [], n_in + 1  # +1: validity mask as an input channel
        for _ in range(3):
            layers += [nn.Conv1d(c, hidden, 5, padding=2), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)]
            c = hidden
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(2 * hidden, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, x, m, doy):
        h = self.conv(torch.cat([x, m.unsqueeze(-1).float()], -1).transpose(1, 2))   # [B, H, T]
        mf = m.unsqueeze(1).float()
        avg = (h * mf).sum(-1) / mf.sum(-1).clamp(min=1)
        mx = h.masked_fill(~m.unsqueeze(1), -1e4).amax(-1)
        mx = torch.where(m.any(-1, keepdim=True), mx, torch.zeros_like(mx))
        return self.head(torch.cat([avg, mx], -1))


def doy_encoding(doy: torch.Tensor, d: int) -> torch.Tensor:
    i = torch.arange(d // 2, device=doy.device, dtype=torch.float32)
    freq = 1.0 / (1000 ** (2 * i / d))
    ang = doy.unsqueeze(-1) * freq
    return torch.cat([torch.sin(ang), torch.cos(ang)], -1)


class LTAELite(nn.Module):
    """Lightweight Temporal Attention Encoder (Garnot & Landrieu, 2020), per pixel.

    Learned master query per head; invalid dates are masked out of the attention, which handles
    irregular, cloud-gapped series without interpolation. Positions are encoded by day of year.
    """

    def __init__(self, n_in: int, n_classes: int, d: int = 64, n_head: int = 4, d_k: int = 8, dropout: float = 0.2):
        super().__init__()
        self.d, self.n_head, self.d_k = d, n_head, d_k
        self.inp = nn.Sequential(nn.Linear(n_in, d), nn.LayerNorm(d))
        self.key = nn.Linear(d, n_head * d_k)
        self.query = nn.Parameter(torch.randn(n_head, d_k) * 0.1)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 128), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(128, n_classes))

    def forward(self, x, m, doy):
        B, T, _ = x.shape
        h = self.inp(x) + doy_encoding(doy, self.d)                   # [B, T, d]
        k = self.key(h).view(B, T, self.n_head, self.d_k)
        att = torch.einsum("bthk,hk->bht", k, self.query) / math.sqrt(self.d_k)
        mm = m.clone()
        mm[~mm.any(-1)] = True                                        # all-cloudy pixel: attend uniformly
        att = att.masked_fill(~mm.unsqueeze(1), -1e4).softmax(-1)
        v = h.view(B, T, self.n_head, self.d // self.n_head)
        z = torch.einsum("bht,bthc->bhc", att, v).reshape(B, self.d)
        return self.out(z)


def class_alpha(y: np.ndarray, n_classes: int) -> np.ndarray:
    cnt = np.bincount(y, minlength=n_classes).astype(float)
    a = np.where(cnt > 0, 1 / np.sqrt(np.maximum(cnt, 1)), 0.0)
    return (a / a[cnt > 0].mean()).astype(np.float32)


def focal_loss(logits, y, alpha, gamma: float):
    logp = F.log_softmax(logits, -1).gather(1, y[:, None]).squeeze(1)
    p = logp.exp()
    return (-alpha[y] * (1 - p) ** gamma * logp).mean()


def make_sequence(name: str, n_in: int, n_classes: int) -> nn.Module:
    return {"tempcnn": TempCNN, "ltae": LTAELite}[name](n_in, n_classes)


def train_sequence(name: str, Z: np.ndarray, M: np.ndarray, doy: np.ndarray, y: np.ndarray, n_classes: int,
                   cfg: dict, seed: int, val_frac: float = 0.1, log=None) -> nn.Module:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_sequence(name, Z.shape[-1], n_classes)
    alpha = torch.tensor(class_alpha(y, n_classes))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["modelos"]["lr"], weight_decay=1e-4)
    epochs, bs = cfg["run"]["epocas"], cfg["modelos"]["batch"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["modelos"]["lr"], total_steps=epochs * math.ceil(len(y) / bs))
    Zt, Mt, Dt, yt = map(torch.from_numpy, (Z, M, doy.astype(np.float32), y.astype(np.int64)))
    gamma = cfg["modelos"]["focal_gamma"]
    for ep in range(epochs):
        model.train()
        perm = torch.from_numpy(rng.permutation(len(y)))
        tot = 0.0
        for i in range(0, len(y), bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            x = Zt[b]
            # augmentation: randomly drop 10% of valid dates (simulates extra clouds)
            mb = Mt[b] & (torch.rand(Mt[b].shape) > 0.1)
            x = x * mb.unsqueeze(-1)
            loss = focal_loss(model(x, mb, Dt[b] if Dt.ndim == 2 else Dt.expand(len(b), -1)), yt[b], alpha, gamma)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        if log:
            log(f"  {name} época {ep + 1}/{epochs} perda={tot / len(y):.4f}")
    model.eval()
    return model


class SmallUNet(nn.Module):
    """Three-level U-Net for binary road segmentation on temporal composites (optional, full mode)."""

    def __init__(self, n_in: int, base: int = 16):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())

        self.e1, self.e2, self.e3 = blk(n_in, base), blk(base, 2 * base), blk(2 * base, 4 * base)
        self.mid = blk(4 * base, 8 * base)
        self.u3, self.d3 = nn.ConvTranspose2d(8 * base, 4 * base, 2, 2), blk(8 * base, 4 * base)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * base, 2 * base, 2, 2), blk(4 * base, 2 * base)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * base, base, 2, 2), blk(2 * base, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        m = self.mid(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(m), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


@torch.no_grad()
def sequence_proba(model: nn.Module, Z: np.ndarray, M: np.ndarray, doy: np.ndarray, chunk: int = 65536) -> np.ndarray:
    out = []
    D = torch.from_numpy(doy.astype(np.float32))
    for i in range(0, len(Z), chunk):
        z, m = torch.from_numpy(Z[i:i + chunk]), torch.from_numpy(M[i:i + chunk])
        d = D[i:i + chunk] if D.ndim == 2 else D.expand(len(z), -1)
        out.append(model(z, m, d).softmax(-1).numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)
