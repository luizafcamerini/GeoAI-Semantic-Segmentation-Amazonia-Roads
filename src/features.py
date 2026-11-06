"""Per-pixel time-series features: spectral indices, temporal statistics, Fourier harmonics."""
from __future__ import annotations

import datetime as dt
import warnings

import numpy as np

INDICES = {
    # name: (a, b) -> (a - b) / (a + b)
    "ndvi": ("nir", "red"),
    "ndwi": ("green", "nir"),      # McFeeters (1996)
    "mndwi": ("green", "swir1"),   # Xu (2006), needs SWIR
    "ndbi": ("swir1", "nir"),      # Zha et al. (2003), needs SWIR
}


def variable_names(bands) -> list[str]:
    bands = [str(b) for b in bands]
    return bands + [k for k, (a, b) in INDICES.items() if a in bands and b in bands]


def series_for_pixels(cube: dict, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Return X [n, T, V] float32 with NaN where the observation is invalid (cloud/no data)."""
    bands = [str(b) for b in cube["bands"]]
    d = cube["data"][:, :, rows, cols].astype(np.float32) / 1e4      # [T, C, n]
    v = cube["valid"][:, rows, cols]                                  # [T, n]
    d = np.where(v[:, None, :], d, np.nan)
    feats = [d[:, i] for i in range(len(bands))]
    with np.errstate(divide="ignore", invalid="ignore"):
        for k, (a, b) in INDICES.items():
            if a in bands and b in bands:
                A, B = d[:, bands.index(a)], d[:, bands.index(b)]
                feats.append(np.clip((A - B) / (A + B), -1, 1))
    X = np.stack(feats, axis=-1)                                      # [T, n, V]
    return np.ascontiguousarray(X.transpose(1, 0, 2))


def day_of_year(dates) -> np.ndarray:
    return np.array([dt.date.fromisoformat(str(d)).timetuple().tm_yday for d in dates], np.float32)


def interp_time(X: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaN gaps along time (axis 1); edges get nearest value; all-NaN stays NaN."""
    n, T, V = X.shape
    t = np.arange(T)
    ok = np.isfinite(X)
    # index of previous / next valid observation
    prev = np.where(ok, t[None, :, None], -1)
    prev = np.maximum.accumulate(prev, axis=1)
    nxt = np.where(ok, t[None, :, None], T)
    nxt = np.minimum.accumulate(nxt[:, ::-1], axis=1)[:, ::-1]
    p = np.clip(prev, 0, T - 1)
    q = np.clip(nxt, 0, T - 1)
    Xp = np.take_along_axis(X, p, axis=1)
    Xq = np.take_along_axis(X, q, axis=1)
    has_p, has_q = prev >= 0, nxt < T
    w = np.where(has_p & has_q, (t[None, :, None] - prev) / np.maximum(nxt - prev, 1), 0.0)
    out = np.where(has_p & has_q, Xp + w * (Xq - Xp), np.where(has_p, Xp, Xq))
    return np.where(ok, X, out).astype(np.float32)


def nan_percentiles(X: np.ndarray, qs, axis: int = 1) -> list[np.ndarray]:
    """Vectorized NaN-aware percentiles (linear interpolation), much faster than np.nanpercentile."""
    S = np.sort(X, axis=axis)                      # NaNs sort to the end
    n = np.isfinite(X).sum(axis=axis, keepdims=True)
    out = []
    for q in qs:
        pos = (q / 100.0) * np.maximum(n - 1, 0)
        lo = np.floor(pos).astype(np.int64)
        hi = np.minimum(lo + 1, np.maximum(n - 1, 0))
        a = np.take_along_axis(S, lo, axis=axis)
        b = np.take_along_axis(S, hi, axis=axis)
        v = a + (pos - lo) * (b - a)
        out.append(np.where(n > 0, v, np.nan).squeeze(axis))
    return out


def harmonic_design(doy: np.ndarray, n_harm: int = 2) -> np.ndarray:
    tt = 2 * np.pi * doy / 365.25
    cols = [np.ones_like(tt)]
    for k in range(1, n_harm + 1):
        cols += [np.cos(k * tt), np.sin(k * tt)]
    return np.stack(cols, 1)


def aggregate_features(X: np.ndarray, var_names: list[str], doy: np.ndarray,
                       harm_vars=("ndvi", "ndwi", "mndwi")) -> tuple[np.ndarray, list[str]]:
    """Temporal statistics per variable + Fourier harmonics (fenologia) + number of valid observations."""
    feats, names = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(X, 1)
        std = np.nanstd(X, 1)
        p10, p50, p90 = nan_percentiles(X, [10, 50, 90], axis=1)
    for arr, tag in ((mean, "media"), (std, "desvio"), (p10, "p10"), (p50, "p50"), (p90, "p90"), (p90 - p10, "amplitude")):
        feats.append(arr)
        names += [f"{v}_{tag}" for v in var_names]
    Xi = interp_time(X)
    A = harmonic_design(doy)
    pinv = np.linalg.pinv(A)                                          # [2h+1, T]
    for v in harm_vars:
        if v not in var_names:
            continue
        s = Xi[:, :, var_names.index(v)]
        s = np.where(np.isfinite(s), s, 0.0)
        coef = s @ pinv.T                                             # [n, 2h+1]
        a1 = np.hypot(coef[:, 1], coef[:, 2])
        ph1 = np.arctan2(coef[:, 2], coef[:, 1])
        a2 = np.hypot(coef[:, 3], coef[:, 4])
        feats.append(np.stack([coef[:, 0], a1, np.cos(ph1), np.sin(ph1), a2], 1))
        names += [f"{v}_harm_{k}" for k in ("c0", "amp1", "cosfase1", "senfase1", "amp2")]
    nvalid = np.isfinite(X).all(-1).sum(1, keepdims=True).astype(np.float32)
    feats.append(nvalid)
    names.append("n_obs_validas")
    F = np.concatenate(feats, 1).astype(np.float32)
    return np.nan_to_num(F, nan=0.0), names


def valid_obs_count(cube: dict) -> np.ndarray:
    return cube["valid"].sum(0)


# --------------------------------------------------------------------------------------
# deep-model inputs
# --------------------------------------------------------------------------------------
def norm_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(X, axis=(0, 1))
        sd = np.nanstd(X, axis=(0, 1))
    return np.nan_to_num(mu), np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)


def to_sequence(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalized sequence with zeros at invalid dates, plus the validity mask [n, T]."""
    M = np.isfinite(X).all(-1)
    Z = (X - mu) / sd
    Z = np.where(M[..., None], Z, 0.0).astype(np.float32)
    return Z, M
