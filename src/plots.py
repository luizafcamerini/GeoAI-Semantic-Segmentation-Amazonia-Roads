"""Plot helpers shared by the notebook."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

CLASS_COLORS = {
    "outros": "#d9d3c7",
    "agua": "#2b6cb0",
    "vegetacao_baixa": "#c6d96a",
    "floresta": "#1f6b3a",
    "construcoes": "#b83232",
    "estrada": "#111111",
}
CLASS_LABELS = {"outros": "outros", "agua": "água", "vegetacao_baixa": "vegetação baixa", "floresta": "floresta",
                "construcoes": "construções", "estrada": "estrada"}
MODEL_COLORS = {"rf": "#4c78a8", "lgbm": "#f58518", "tempcnn": "#54a24b", "ltae": "#b279a2", "unet": "#9d755d"}
MODEL_LABELS = {"rf": "Random Forest", "lgbm": "LightGBM", "tempcnn": "TempCNN", "ltae": "L-TAE leve", "unet": "U-Net"}


def class_cmap(names):
    cmap = ListedColormap([CLASS_COLORS[n] for n in names])
    cmap.set_bad("white")
    return cmap


def show_classes(ax, arr, names, title=""):
    a = np.ma.masked_equal(arr, 255)
    ax.imshow(a, cmap=class_cmap(names), vmin=-0.5, vmax=len(names) - 0.5, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()


def class_legend(fig, names, **kw):
    handles = [Patch(color=CLASS_COLORS[n], label=CLASS_LABELS[n]) for n in names]
    fig.legend(handles=handles, loc="lower center", ncol=len(names), frameon=False, fontsize=9, **kw)


def error_map(label, pred):
    """0 = correct, 1 = error, 255 = not evaluated."""
    out = np.full(label.shape, 255, np.uint8)
    ok = (label != 255) & (pred != 255)
    out[ok] = (label[ok] != pred[ok]).astype(np.uint8)
    return out


def f1_vs_resolution(ax, df, metric="f1_estrada", title="", show_cbers=True):
    """One line per model (S2 native 10 m + degraded 30/64 m); CBERS 64 m real as separate markers."""
    s2 = df[df["sensor"] == "s2"]
    for m, g in s2.groupby("modelo"):
        c = MODEL_COLORS.get(m, "gray")
        single = g.groupby("res_m").size().max() == 1          # one AOI per point: its block-bootstrap CI is valid
        a = g.groupby("res_m")[metric].mean()
        ax.plot(a.index, a.values, "-o", color=c, label=MODEL_LABELS.get(m, m), lw=2)
        if single and f"{metric}_ic_inf" in g:
            gg = g.sort_values("res_m")
            ax.fill_between(gg["res_m"], gg[f"{metric}_ic_inf"], gg[f"{metric}_ic_sup"], color=c, alpha=0.12, lw=0)
        elif not single:                                         # several AOIs: show the spread across AOIs
            lo, hi = g.groupby("res_m")[metric].min(), g.groupby("res_m")[metric].max()
            ax.fill_between(lo.index, lo.values, hi.values, color=c, alpha=0.08, lw=0)
    if show_cbers:
        cb = df[df["sensor"] == "cbers"]
        for m, g in cb.groupby("modelo"):
            ax.plot([64 + 2], [g[metric].mean()], marker="D", ls="none", color=MODEL_COLORS.get(m, "gray"),
                    markeredgecolor="black", markersize=7)
        if len(cb):
            ax.plot([], [], marker="D", ls="none", color="white", markeredgecolor="black", label="CBERS-4/4A WFI 64 m (real)")
    ax.set_xticks([10, 30, 64])
    ax.set_xlabel("resolução espacial (m)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, frameon=False)
