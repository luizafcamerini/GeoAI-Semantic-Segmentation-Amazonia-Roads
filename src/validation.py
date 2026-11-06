"""Spatial validation design: blocks with buffer, geographic holdout, leave-one-region-out, transfer.

Blocks are defined in metres, so the same fold layout is used at 10, 30 and 64 m.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import maximum_filter

IGNORE = 255


def block_of_pixel(shape: tuple[int, int], res: int, bloco_m: int) -> np.ndarray:
    """Block id of every pixel (row-major over the block lattice)."""
    H, W = shape
    nb = math.ceil(W * res / bloco_m)
    by = (np.arange(H) * res) // bloco_m
    bx = (np.arange(W) * res) // bloco_m
    return (by[:, None] * nb + bx[None, :]).astype(np.int32)


def assign_block_folds(size_m: int, bloco_m: int, k: int, seed: int) -> np.ndarray:
    """Random, balanced fold id for each block (shape [n_blocks])."""
    nb = math.ceil(size_m / bloco_m)
    rng = np.random.default_rng(seed)
    folds = np.arange(nb * nb) % k
    rng.shuffle(folds)
    return folds


def fold_masks(shape, res: int, bloco_m: int, buffer_m: int, block_folds: np.ndarray, fold: int):
    """Train/test masks for one fold. Train excludes a `buffer_m` margin around every test block."""
    blk = block_of_pixel(shape, res, bloco_m)
    test = block_folds[blk] == fold
    b = max(1, math.ceil(buffer_m / res))
    near_test = maximum_filter(test, size=2 * b + 1)
    train = ~near_test
    return train, test


def leakage_report(train: np.ndarray, test: np.ndarray, res: int, buffer_m: int) -> dict:
    """Checks: no shared pixel; no train pixel within buffer of a test pixel (Chebyshev distance)."""
    b = max(1, math.ceil(buffer_m / res))
    near = maximum_filter(test, size=2 * b + 1)
    return {"pixels_compartilhados": int((train & test).sum()),
            "treino_dentro_do_buffer": int((train & near).sum()),
            "n_treino": int(train.sum()), "n_teste": int(test.sum())}


def design_experiments(cfg: dict) -> list[dict]:
    """All experiments. Every *test* AOI is Brazilian; the holdout AOI is never in a training set."""
    aois = cfg["run"]["aois"]
    br = [a for a in aois if cfg["aois"][a]["pais"] == "BR"]
    holdout = [a for a in br if cfg["aois"][a]["papel"] == "holdout"]
    br_train = [a for a in br if cfg["aois"][a]["papel"] == "treino"]
    eu = [a for a in aois if cfg["aois"][a]["papel"] == "transferencia"]
    k = cfg["run"]["folds_blocos"]
    exps = []
    for a in br_train:
        for f in range(k):
            exps.append({"tipo": "blocos", "nome": f"blocos:{a}:f{f}", "treino": [a], "teste": [a], "fold": f})
    exps.append({"tipo": "holdout", "nome": "holdout", "treino": list(br_train), "teste": list(holdout)})
    for a in br_train:
        exps.append({"tipo": "loro", "nome": f"loro:{a}", "treino": [b for b in br_train if b != a], "teste": [a]})
    if eu:
        exps.append({"tipo": "transferencia", "nome": "transferencia:EU->BR", "treino": list(eu),
                     "teste": list(br_train) + list(holdout)})
    check_design(exps, cfg)
    return exps


def check_design(exps: list[dict], cfg: dict) -> None:
    holdout = {a for a in cfg["aois"] if cfg["aois"][a]["papel"] == "holdout"}
    for e in exps:
        assert all(cfg["aois"][a]["pais"] == "BR" for a in e["teste"]), f"teste fora do Brasil em {e['nome']}"
        assert not holdout & set(e["treino"]), f"holdout usado no treino em {e['nome']}"
        if e["tipo"] != "blocos":
            assert not set(e["treino"]) & set(e["teste"]), f"AOI em treino e teste em {e['nome']}"


def sample_pixels(labels: np.ndarray, mask: np.ndarray, n: int, n_classes: int, seed: int):
    """Class-capped random sample of (row, col) inside `mask`; rare classes (roads) are kept whole up to the cap."""
    rng = np.random.default_rng(seed)
    ok = mask & (labels != IGNORE)
    cap = n // n_classes
    picks = []
    for c in range(n_classes):
        idx = np.flatnonzero(ok & (labels == c))
        if len(idx):
            picks.append(rng.choice(idx, min(cap, len(idx)), replace=False))
    sel = np.concatenate(picks) if picks else np.array([], int)
    rest = n - len(sel)
    if rest > 0:
        pool = np.setdiff1d(np.flatnonzero(ok), sel, assume_unique=False)
        if len(pool):
            sel = np.concatenate([sel, rng.choice(pool, min(rest, len(pool)), replace=False)])
    rng.shuffle(sel)
    return np.unravel_index(sel, labels.shape)
