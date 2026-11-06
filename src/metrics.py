"""Metrics: per-class scores, spatial-block bootstrap CIs, relaxed and length-based road metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, convolve
from skimage.morphology import skeletonize

IGNORE = 255


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(y_true.astype(int) * n + y_pred.astype(int), minlength=n * n).reshape(n, n)


def scores_from_cm(cm: np.ndarray, names: list[str]) -> dict:
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        rec = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        f1 = np.where(np.isfinite(prec) & np.isfinite(rec) & (prec + rec > 0), 2 * prec * rec / (prec + rec), 0.0)
        f1 = np.where(tp + fn > 0, f1, np.nan)      # class absent from reference: undefined
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), np.nan)
    out = {"acuracia_global": float(tp.sum() / max(cm.sum(), 1)), "f1_macro": float(np.nanmean(f1)),
           "n_pixels": int(cm.sum())}
    for i, n in enumerate(names):
        out[f"precisao_{n}"], out[f"revocacao_{n}"] = float(prec[i]), float(rec[i])
        out[f"f1_{n}"], out[f"iou_{n}"] = float(f1[i]), float(iou[i])
    return out


def block_confusions(y_true, y_pred, blocks, n) -> dict[int, np.ndarray]:
    out = {}
    order = np.argsort(blocks, kind="stable")
    b_sorted = blocks[order]
    cuts = np.flatnonzero(np.diff(b_sorted)) + 1
    for idx in np.split(order, cuts):
        if len(idx):
            out[int(blocks[idx[0]])] = confusion(y_true[idx], y_pred[idx], n)
    return out


def bootstrap_ci(block_cms: dict[int, np.ndarray], names: list[str], keys: list[str], n_boot: int, seed: int,
                 alpha: float = 0.05) -> dict:
    """Percentile CIs by resampling spatial blocks (not pixels: pixels are autocorrelated)."""
    rng = np.random.default_rng(seed)
    cms = np.stack(list(block_cms.values()))
    vals = {k: [] for k in keys}
    for _ in range(n_boot):
        s = scores_from_cm(cms[rng.integers(0, len(cms), len(cms))].sum(0), names)
        for k in keys:
            vals[k].append(s[k])
    out = {}
    for k, v in vals.items():
        v = np.array(v, float)
        v = v[np.isfinite(v)]
        out[f"{k}_ic_inf"] = float(np.quantile(v, alpha / 2)) if len(v) else np.nan
        out[f"{k}_ic_sup"] = float(np.quantile(v, 1 - alpha / 2)) if len(v) else np.nan
    return out


# --------------------------------------------------------------------------------------
# roads: relaxed (buffered) precision/recall and length-based completeness/correctness
# --------------------------------------------------------------------------------------
def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def relaxed_pr(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray, tol_px: int) -> dict:
    """Relaxed precision: predicted road pixels within `tol_px` of a reference road; relaxed recall: the converse."""
    gt, pred = gt & valid, pred & valid
    k = _disk(tol_px)
    p = (pred & binary_dilation(gt, k)).sum() / max(pred.sum(), 1)
    r = (gt & binary_dilation(pred, k)).sum() / max(gt.sum(), 1)
    f = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {f"precisao_relax{tol_px}": float(p), f"revocacao_relax{tol_px}": float(r), f"f1_relax{tol_px}": float(f)}


def skeleton_length_per_pixel(skel: np.ndarray, res: float) -> np.ndarray:
    """Length (m) carried by each skeleton pixel: half of each 4-/8-neighbour edge (1 or sqrt 2 px)."""
    s = skel.astype(np.float32)
    k4 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32)
    kd = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], np.float32)
    n4 = convolve(s, k4, mode="constant")
    nd = convolve(s, kd, mode="constant")
    length = 0.5 * (n4 + np.sqrt(2) * nd) * res
    length[(n4 + nd) == 0] = res  # isolated pixel: count one pixel length
    return np.where(skel, length, 0.0)


def length_metrics(gt: np.ndarray, pred: np.ndarray, valid: np.ndarray, res: float, tol_px: int = 2) -> dict:
    """Completeness/correctness by length of skeletonized networks (Wiedemann et al. 1998 / Imazon logic)."""
    sg = skeletonize(gt & valid)
    sp = skeletonize(pred & valid)
    lg, lp = skeleton_length_per_pixel(sg, res), skeleton_length_per_pixel(sp, res)
    k = _disk(tol_px)
    comp = lg[binary_dilation(sp, k)].sum() / max(lg.sum(), 1e-9)
    corr = lp[binary_dilation(sg, k)].sum() / max(lp.sum(), 1e-9)
    return {"comprimento_ref_km": float(lg.sum() / 1e3), "comprimento_pred_km": float(lp.sum() / 1e3),
            "completude": float(comp), "correcao": float(corr),
            "qualidade": float(comp * corr / max(comp + corr - comp * corr, 1e-9))}


def skeleton_to_lines(skel: np.ndarray, transform) -> "list":
    """Vectorize a skeleton: every pair of 8-connected pixels becomes a segment, merged into lines."""
    from shapely.geometry import LineString, MultiLineString
    from shapely.ops import linemerge

    r, c = np.nonzero(skel)
    on = set(zip(r.tolist(), c.tolist()))
    segs = []
    for (i, j) in on:
        for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
            if (i + di, j + dj) in on:
                x0, y0 = transform * (j + 0.5, i + 0.5)
                x1, y1 = transform * (j + dj + 0.5, i + di + 0.5)
                segs.append(LineString([(x0, y0), (x1, y1)]))
    if not segs:
        return []
    merged = linemerge(MultiLineString(segs))
    return list(merged.geoms) if hasattr(merged, "geoms") else [merged]


# --------------------------------------------------------------------------------------
# one-stop evaluation of a prediction map
# --------------------------------------------------------------------------------------
def evaluate_map(label: np.ndarray, pred: np.ndarray, eval_mask: np.ndarray, blocks: np.ndarray, names: list[str],
                 res: int, road_id: int, n_boot: int, seed: int, tol_px=(1, 2)) -> dict:
    m = eval_mask & (label != IGNORE) & (pred != IGNORE)
    y, p, b = label[m], pred[m], blocks[m]
    n = len(names)
    cm = confusion(y, p, n)
    out = scores_from_cm(cm, names)
    out["matriz_confusao"] = cm
    if n_boot and len(np.unique(b)) > 1:
        out.update(bootstrap_ci(block_confusions(y, p, b, n), names,
                                ["f1_macro", "acuracia_global", "f1_estrada", "iou_estrada"], n_boot, seed))
    gt_r = (label == road_id) & m
    pr_r = (pred == road_id) & m
    for t in tol_px:
        out.update(relaxed_pr(gt_r, pr_r, m, t))
    out.update(length_metrics(gt_r, pr_r, m, res, tol_px=max(tol_px)))
    return out


def per_point_scores(df: pd.DataFrame, pred_col: str, ref_col: str, names: list[str]) -> dict:
    idx = {n: i for i, n in enumerate(names)}
    y = df[ref_col].map(idx).to_numpy()
    p = df[pred_col].map(idx).to_numpy()
    ok = ~(pd.isna(y) | pd.isna(p))
    return scores_from_cm(confusion(y[ok].astype(int), p[ok].astype(int), len(names)), names)
