"""Orchestration: lazy data store (cubes, labels, features) and experiment runner."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import features as FT
from . import labels as L
from . import metrics as MT
from . import models as MD
from . import validation as V
from .config import DATA, RESULTS, aoi_grid, save_json

IGNORE = 255

# data key -> (sensor, resolution, how it is produced)
DATA_KEYS = {
    "s2_10": ("s2", 10, "nativo"),
    "s2_30": ("s2", 30, "degradado"),
    "s2_64": ("s2", 64, "degradado"),
    "cbers_64": ("cbers", 64, "nativo"),
    "landsat_30": ("landsat", 30, "nativo"),
}


def active_keys(cfg: dict) -> list[str]:
    keys = ["s2_10", "s2_30", "s2_64", "cbers_64"]
    if cfg["run"]["usar_landsat"]:
        keys.append("landsat_30")
    return keys


class Store:
    """Lazy, cached access to cubes, labels, and features for every (AOI, data key)."""

    def __init__(self, cfg: dict, log=print):
        self.cfg, self.log = cfg, log
        self._cubes, self._labels, self._feats, self._roadfrac, self._base = {}, {}, {}, {}, {}
        self.collections: dict[str, str] = {}
        self.road_vectors: dict[str, dict] = {}

    def grid(self, aoi: str, res: int):
        return aoi_grid(self.cfg, aoi, res)

    def is_br(self, aoi: str) -> bool:
        return self.cfg["aois"][aoi]["pais"] == "BR"

    # ---------------- cubes ----------------
    def available(self, aoi: str, key: str) -> bool:
        try:
            self.cube(aoi, key)
            return True
        except Exception as e:  # noqa: BLE001
            self.log(f"[indisponível] {aoi} {key}: {e}")
            return False

    def cube(self, aoi: str, key: str) -> dict:
        if (aoi, key) in self._cubes:
            return self._cubes[(aoi, key)]
        sensor, res, how = DATA_KEYS[key]
        if how == "degradado":
            path = D.cube_path(aoi, key)
            if path.exists():
                c = dict(np.load(path, allow_pickle=True))
            else:
                c = D.degrade_cube(self.cube(aoi, "s2_10"), self.grid(aoi, 10), res)
                np.savez_compressed(path, **c)
        elif not self.is_br(aoi):
            if sensor != "s2":
                raise RuntimeError(f"{sensor} não cobre {aoi} (sensor brasileiro / cubo BDC só no Brasil)")
            c = D.load_pc_s2_cube(self.cfg, aoi, self.grid(aoi, res), self.cfg["run"]["max_datas"])
        else:
            coll = self.collections.get(sensor)
            if coll is None:
                raise RuntimeError(f"coleção para {sensor} não definida")
            try:
                c = D.load_bdc_cube(self.cfg, aoi, self.grid(aoi, res), sensor, coll, self.cfg["run"]["max_datas"])
            except Exception as e:  # noqa: BLE001
                if sensor != "s2":
                    raise
                D.log_status(f"BDC:{coll}:{aoi}", False, f"{e}; usando Planetary Computer")
                c = D.load_pc_s2_cube(self.cfg, aoi, self.grid(aoi, res), self.cfg["run"]["max_datas"])
        self._cubes[(aoi, key)] = c
        return c

    def doy(self, aoi: str, key: str) -> np.ndarray:
        return FT.day_of_year(self.cube(aoi, key)["dates"])

    def var_names(self, aoi: str, key: str) -> list[str]:
        return FT.variable_names(self.cube(aoi, key)["bands"])

    # ---------------- labels ----------------
    def base_classes(self, aoi: str, res: int) -> np.ndarray:
        """MapBiomas (Brazil) or ESA WorldCover (Europe) mapped to the study classes, no roads."""
        if (aoi, res) in self._base:
            return self._base[(aoi, res)]
        g = self.grid(aoi, res)
        if self.is_br(aoi):
            raw = D.load_mapbiomas(self.cfg, aoi, g)
            base = L.build_lut(self.cfg["mapbiomas_para_estudo"], self.cfg)[raw]
        else:
            from rasterio.enums import Resampling

            raw = D.load_pc_raster(self.cfg, aoi, g, "esa-worldcover", "map",
                                   resampling=Resampling.nearest if res <= 10 else Resampling.mode)
            raw = np.nan_to_num(raw, nan=0).astype(np.uint8)
            base = L.build_lut(self.cfg["worldcover_para_estudo"], self.cfg)[raw]
        self._base[(aoi, res)] = base
        return base

    def roads(self, aoi: str):
        """Road vectors used for labels: OSM (+ Imazon if ever available). Official sources are cross-checks only."""
        if aoi not in self.road_vectors:
            self.road_vectors[aoi] = {}
            try:
                self.road_vectors[aoi]["OSM"] = D.load_osm_roads(self.cfg, aoi, self.grid(aoi, 10))
            except Exception as e:  # noqa: BLE001
                D.log_status(f"OSM:{aoi}", False, e)
        return self.road_vectors[aoi]

    def road_fraction(self, aoi: str, res: int) -> np.ndarray:
        if (aoi, res) not in self._roadfrac:
            path = DATA / "rotulos" / f"{aoi}__fracao_estrada_{res}.npy"
            if not path.exists():
                rv = self.roads(aoi).get("OSM")
                if rv is None:  # never cache an all-zero road label because a download failed
                    raise RuntimeError(f"estradas OSM indisponíveis para {aoi}; rótulo de estrada não pode ser construído")
                polys = L.buffer_roads(rv, self.cfg) if len(rv) else []
                res_all = sorted({DATA_KEYS[k][1] for k in DATA_KEYS})
                fr = L.road_fractions(polys, self.grid(aoi, 10), res_all, self.cfg["estradas"]["subgrade_m"])
                for r, a in fr.items():
                    np.save(DATA / "rotulos" / f"{aoi}__fracao_estrada_{r}.npy", a)
            self._roadfrac[(aoi, res)] = np.load(path)
        return self._roadfrac[(aoi, res)]

    def labels(self, aoi: str, res: int) -> np.ndarray:
        if (aoi, res) not in self._labels:
            thr = L.road_threshold(self.cfg, res)
            self._labels[(aoi, res)] = L.compose_labels(self.base_classes(aoi, res), self.road_fraction(aoi, res),
                                                        thr, self.cfg)
        return self._labels[(aoi, res)]

    # ---------------- features ----------------
    def features(self, aoi: str, key: str) -> np.ndarray:
        """Aggregated features for every pixel, [H*W, D] (cached on disk)."""
        if (aoi, key) in self._feats:
            return self._feats[(aoi, key)]
        path = DATA / "atributos" / f"{aoi}__{key}.npy"
        if not path.exists():
            cube = self.cube(aoi, key)
            H, W = cube["valid"].shape[1:]
            names = self.var_names(aoi, key)
            doy = self.doy(aoi, key)
            chunks = []
            rows_per = max(1, 120_000 // W)
            for r0 in range(0, H, rows_per):
                rr, cc = np.mgrid[r0:min(H, r0 + rows_per), 0:W]
                X = FT.series_for_pixels(cube, rr.ravel(), cc.ravel())
                F, fn = FT.aggregate_features(X, names, doy)
                chunks.append(F)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.concatenate(chunks))
            save_json(fn, path.with_suffix(".json"))
        self._feats[(aoi, key)] = np.load(path, mmap_mode="r")
        return self._feats[(aoi, key)]

    def feature_names(self, aoi: str, key: str) -> list[str]:
        self.features(aoi, key)
        import json

        return json.loads((DATA / "atributos" / f"{aoi}__{key}.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# experiment runner
# --------------------------------------------------------------------------------------
def _pad_T(X: np.ndarray, doy: np.ndarray, T: int):
    if X.shape[1] == T:
        return X, np.broadcast_to(doy, (len(X), T)).copy()
    pad = T - X.shape[1]
    Xp = np.concatenate([X, np.full((len(X), pad, X.shape[2]), np.nan, np.float32)], 1)
    dp = np.concatenate([doy, np.zeros(pad, np.float32)])
    return Xp, np.broadcast_to(dp, (len(X), T)).copy()


def _masks_for(store: Store, exp: dict, aoi: str, res: int, role: str) -> np.ndarray:
    cfg = store.cfg
    shape = store.grid(aoi, res).shape
    if exp["tipo"] != "blocos":
        return np.ones(shape, bool)
    folds = V.assign_block_folds(cfg["run"]["tamanho_aoi_m"], cfg["run"]["bloco_m"], cfg["run"]["folds_blocos"],
                                 cfg["projeto"]["seed"])
    tr, te = V.fold_masks(shape, res, cfg["run"]["bloco_m"], cfg["run"]["buffer_m"], folds, exp["fold"])
    return tr if role == "treino" else te


def run_experiment(store: Store, exp: dict, model_name: str, key: str, preds: dict, log=print) -> dict:
    """Fit `model_name` on exp['treino'] and fill `preds[aoi]` (uint8 map, 255 = not predicted) for exp['teste']."""
    cfg = store.cfg
    seed = cfg["projeto"]["seed"]
    n_cls = len(cfg["classes"]["nomes"])
    _, res, _ = DATA_KEYS[key]
    t0 = time.time()
    # ---------- training sample ----------
    Xs, Fs, ys, ds, split_log = [], [], [], [], {}
    for a in exp["treino"]:
        lab = store.labels(a, res)
        mask = _masks_for(store, exp, a, res, "treino") & store.cube(a, key)["valid"].any(0)
        rr, cc = V.sample_pixels(lab, mask, cfg["run"]["pixels_treino_por_aoi"], n_cls, seed)
        split_log[a] = np.ravel_multi_index((rr, cc), lab.shape)
        ys.append(lab[rr, cc])
        if model_name in MD.TABULAR:
            Fs.append(np.asarray(store.features(a, key)[split_log[a]]))
        else:
            Xs.append(FT.series_for_pixels(store.cube(a, key), rr, cc))
            ds.append(store.doy(a, key))
    y = np.concatenate(ys).astype(np.int64)
    # save the exact training pixels for the leakage test
    sp = RESULTS / "splits" / f"{exp['nome'].replace(':', '_').replace('>', '')}__{key}.npz"
    sp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sp, **{f"treino__{a}": v for a, v in split_log.items()},
                        **{f"teste__{a}": np.flatnonzero(_masks_for(store, exp, a, res, "teste"))
                           for a in exp["teste"]}, res=res, size_m=cfg["run"]["tamanho_aoi_m"],
                        buffer_m=cfg["run"]["buffer_m"])

    # ---------- fit ----------
    if model_name in MD.TABULAR:
        model = MD.make_tabular(model_name, cfg, seed)
        model.fit(np.concatenate(Fs), y)
    else:
        T = max(x.shape[1] for x in Xs)
        pads = [_pad_T(x, d, T) for x, d in zip(Xs, ds)]
        X = np.concatenate([p[0] for p in pads])
        doy = np.concatenate([p[1] for p in pads])
        mu, sd = FT.norm_stats(X)
        Z, M = FT.to_sequence(X, mu, sd)
        model = MD.train_sequence(model_name, Z, M, doy, y, n_cls, cfg, seed)
    t_fit = time.time() - t0

    # ---------- predict on test AOIs ----------
    for a in exp["teste"]:
        g = store.grid(a, res)
        test = _masks_for(store, exp, a, res, "teste")
        idx = np.flatnonzero(test)
        out = preds.setdefault(a, np.full(g.shape, IGNORE, np.uint8))
        if model_name in MD.TABULAR:
            P = MD.tabular_proba(model, np.asarray(store.features(a, key)[idx]), n_cls)
            out.flat[idx] = P.argmax(1)
        else:
            cube, doy_a = store.cube(a, key), store.doy(a, key)
            for i in range(0, len(idx), 100_000):
                sub = idx[i:i + 100_000]
                rr, cc = np.unravel_index(sub, g.shape)
                Xa, da = _pad_T(FT.series_for_pixels(cube, rr, cc), doy_a, max(len(doy_a), 1))
                Za, Ma = FT.to_sequence(Xa, mu, sd)
                out.flat[sub] = MD.sequence_proba(model, Za, Ma, da).argmax(1)
    log(f"{exp['nome']:<28} {model_name:<8} {key:<9} n_treino={len(y):>6}  ajuste={t_fit:5.1f}s  total={time.time() - t0:5.1f}s")
    return {"modelo": model, "t_ajuste": t_fit}


def evaluate_predictions(store: Store, preds: dict, tipo: str, model_name: str, key: str) -> list[dict]:
    cfg = store.cfg
    names = cfg["classes"]["nomes"]
    _, res, _ = DATA_KEYS[key]
    rows = []
    for a, pm in preds.items():
        assert store.is_br(a), f"métrica fora do Brasil: {a}"
        lab = store.labels(a, res)
        blocks = V.block_of_pixel(pm.shape, res, cfg["run"]["bloco_m"])
        r = MT.evaluate_map(lab, pm, pm != IGNORE, blocks, names, res, cfg["classes"]["id"]["estrada"],
                            cfg["run"]["bootstrap_n"], cfg["projeto"]["seed"], cfg["estradas"]["tolerancia_relaxada_px"])
        cm = r.pop("matriz_confusao")
        rows.append({"experimento": tipo, "modelo": model_name, "dados": key, "sensor": DATA_KEYS[key][0],
                     "res_m": res, "origem": DATA_KEYS[key][2], "aoi": a, "pais": cfg["aois"][a]["pais"],
                     **r, "_cm": cm})
    return rows


def unet_road_experiment(store: Store, key: str, train_aois: list[str], test_aois: list[str], log=print) -> dict:
    """Optional spatial model for roads only: small U-Net on temporal composites (p10/p50/p90)."""
    import torch

    cfg = store.cfg
    seed = cfg["projeto"]["seed"]
    _, res, _ = DATA_KEYS[key]
    road = cfg["classes"]["id"]["estrada"]

    def stack(a):
        F = store.features(a, key)
        names = store.feature_names(a, key)
        sel = [i for i, n in enumerate(names) if n.endswith(("_p10", "_p50", "_p90"))]
        H, W = store.grid(a, res).shape
        img = np.asarray(F[:, sel]).reshape(H, W, -1).transpose(2, 0, 1)
        return img

    imgs = {a: stack(a) for a in train_aois + test_aois}
    mu = np.mean([imgs[a].mean((1, 2)) for a in train_aois], 0)[:, None, None]
    sd = np.mean([imgs[a].std((1, 2)) for a in train_aois], 0)[:, None, None] + 1e-6
    P = 64
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for a in train_aois:
        im = (imgs[a] - mu) / sd
        lab = store.labels(a, res)
        H, W = lab.shape
        for _ in range(400 if cfg["projeto"]["modo_rapido"] else 2000):
            i, j = rng.integers(0, H - P + 1), rng.integers(0, W - P + 1)
            xs.append(im[:, i:i + P, j:j + P])
            ys.append((lab[i:i + P, j:j + P] == road).astype(np.float32))
    X = torch.tensor(np.stack(xs), dtype=torch.float32)
    Y = torch.tensor(np.stack(ys))[:, None]
    torch.manual_seed(seed)
    net = MD.SmallUNet(X.shape[1])
    pos = float(Y.mean())
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-4)))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    for ep in range(cfg["run"]["epocas"]):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 16):
            b = perm[i:i + 16]
            loss = lossf(net(X[b]), Y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    preds = {}
    with torch.no_grad():
        for a in test_aois:
            im = torch.tensor((imgs[a] - mu) / sd, dtype=torch.float32)[None]
            H, W = im.shape[-2:]
            ph, pw = (-H) % 16, (-W) % 16
            im = torch.nn.functional.pad(im, (0, pw, 0, ph), mode="reflect")
            prob = torch.sigmoid(net(im))[0, 0, :H, :W].numpy()
            base = np.full((H, W), 0, np.uint8)
            base[prob >= 0.5] = road
            preds[a] = base
    log(f"U-Net estradas {key}: treino {train_aois} -> teste {test_aois}")
    return preds
