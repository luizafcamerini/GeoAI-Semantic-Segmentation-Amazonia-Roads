"""Label construction: class mapping, sub-pixel road fractions, cross-source agreement, gold sample."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from affine import Affine
from rasterio.features import rasterize

from .config import ROOT, Grid

IGNORE = 255


# --------------------------------------------------------------------------------------
# class mapping
# --------------------------------------------------------------------------------------
def build_lut(mapping: dict, cfg: dict, default: str = "outros") -> np.ndarray:
    ids = cfg["classes"]["id"]
    lut = np.full(256, ids[default], np.uint8)
    for code, name in mapping.items():
        lut[int(code)] = IGNORE if name == "ignorar" else ids[name]
    return lut


def mapping_table(cfg: dict) -> pd.DataFrame:
    legend = {  # MapBiomas Collection 10 legend (pt-BR)
        3: "Formação Florestal", 5: "Mangue", 6: "Floresta Alagável", 49: "Restinga Arbórea",
        4: "Formação Savânica", 11: "Campo Alagado e Área Pantanosa", 12: "Formação Campestre",
        13: "Outras Formações não Florestais", 50: "Restinga Herbácea", 15: "Pastagem", 24: "Área Urbanizada",
        33: "Rio, Lago e Oceano", 31: "Aquicultura", 9: "Silvicultura", 21: "Mosaico de Usos", 18: "Agricultura",
        19: "Lavoura Temporária", 39: "Soja", 20: "Cana", 40: "Arroz", 62: "Algodão",
        41: "Outras Lavouras Temporárias", 36: "Lavoura Perene", 46: "Café", 47: "Citrus", 35: "Dendê",
        48: "Outras Lavouras Perenes", 23: "Praia, Duna e Areal", 25: "Outras Áreas não Vegetadas",
        29: "Afloramento Rochoso", 30: "Mineração", 32: "Apicum", 75: "Usina Fotovoltaica",
        27: "Não Observado", 0: "Sem dado",
    }
    rows = [{"codigo_mapbiomas": int(k), "classe_mapbiomas": legend.get(int(k), "?"), "classe_estudo": v}
            for k, v in cfg["mapbiomas_para_estudo"].items()]
    rows.append({"codigo_mapbiomas": "-", "classe_mapbiomas": "vias (OSM/BC250/DNIT), fração >= limiar",
                 "classe_estudo": "estrada (sobrepõe a classe MapBiomas)"})
    return pd.DataFrame(rows).sort_values("classe_estudo", kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# thin objects: sub-pixel road coverage fraction
# --------------------------------------------------------------------------------------
def buffer_roads(gdf, cfg: dict, official: bool = False):
    widths = cfg["estradas"]["largura_m"]
    if official or "highway" not in gdf.columns:
        w = np.full(len(gdf), cfg["estradas"]["largura_oficial_m"], float)
    else:
        hw = gdf["highway"].astype(str).str.replace("_link", "", regex=False)
        w = hw.map(lambda h: widths.get(h, widths["padrao"])).to_numpy(float)
    return gdf.geometry.buffer(w / 2, cap_style="flat")


def road_fractions(polys, grid10: Grid, resolutions: list[int], sub_m: int = 2) -> dict[int, np.ndarray]:
    """Fraction of each pixel covered by buffered road polygons, at several resolutions.

    Rasterizes on a `sub_m` super-sampled grid in 960 m strips (960 is a multiple of every
    study resolution, so strips never split a coarse pixel) and averages into each resolution.
    """
    import geopandas as gpd

    from .config import GRID_UNIT_M

    size = grid10.size_m
    out = {r: np.zeros((size // r, size // r), np.float32) for r in resolutions}
    if len(polys) == 0:
        return out
    gs = gpd.GeoSeries(polys).reset_index(drop=True)
    sidx = gs.sindex
    n_fine = size // sub_m
    rows_strip = GRID_UNIT_M // sub_m
    for k in range(size // GRID_UNIT_M):
        maxy = grid10.maxy - k * GRID_UNIT_M
        box = (grid10.minx, maxy - GRID_UNIT_M, grid10.minx + size, maxy)
        hits = sidx.query(shapely.box(*box))
        if len(hits) == 0:
            continue
        tr = Affine(sub_m, 0, grid10.minx, 0, -sub_m, maxy)
        fine = rasterize(((g, 1) for g in gs.iloc[hits]), out_shape=(rows_strip, n_fine), transform=tr,
                         fill=0, dtype="uint8", all_touched=False)
        for r in resolutions:
            f = r // sub_m
            assert f * sub_m == r, (r, sub_m)
            block = fine.reshape(rows_strip // f, f, n_fine // f, f).mean(axis=(1, 3))
            r0 = k * GRID_UNIT_M // r
            out[r][r0:r0 + block.shape[0]] = block
    return out


def road_threshold(cfg: dict, res: int, fator: float | None = None) -> float:
    e = cfg["estradas"]
    f = e["fator_limiar"] if fator is None else fator
    return float(np.clip(f * e["largura_referencia_m"] / res, 0.02, 0.9))


def compose_labels(base: np.ndarray, road_frac: np.ndarray, thr: float, cfg: dict) -> np.ndarray:
    lab = base.copy()
    lab[road_frac >= thr] = cfg["classes"]["id"]["estrada"]
    return lab


def threshold_sensitivity(road_frac: np.ndarray, cfg: dict, res: int) -> pd.DataFrame:
    any_road = road_frac > 0
    rows = []
    for f in cfg["estradas"]["fatores_sensibilidade"]:
        t = road_threshold(cfg, res, f)
        n = int((road_frac >= t).sum())
        rows.append({"res_m": res, "fator": f, "limiar": round(t, 3), "pixels_estrada": n,
                     "pct_area": 100 * n / road_frac.size,
                     "pct_pixels_tocados_mantidos": 100 * n / max(int(any_road.sum()), 1)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# agreement between sources
# --------------------------------------------------------------------------------------
def class_agreement(a: np.ndarray, b: np.ndarray, names: list[str]) -> dict:
    from sklearn.metrics import cohen_kappa_score

    m = (a != IGNORE) & (b != IGNORE)
    x, y = a[m].ravel(), b[m].ravel()
    res = {"concordancia_global": float((x == y).mean()), "kappa": float(cohen_kappa_score(x, y))}
    for i, n in enumerate(names):
        inter = np.sum((x == i) & (y == i))
        union = np.sum((x == i) | (y == i))
        if union:
            res[f"iou_{n}"] = float(inter / union)
    return res


def line_agreement(a, b, tol_m: float = 30.0) -> dict:
    """Length-based agreement: share of A's length within `tol_m` of B, and vice-versa."""
    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return {"comprimento_a_km": float(a.length.sum() / 1e3) if a is not None and len(a) else 0.0,
                "comprimento_b_km": float(b.length.sum() / 1e3) if b is not None and len(b) else 0.0,
                "a_em_b": np.nan, "b_em_a": np.nan}
    ua, ub = a.union_all(), b.union_all()
    la, lb = ua.length, ub.length
    return {"comprimento_a_km": la / 1e3, "comprimento_b_km": lb / 1e3,
            "a_em_b": ua.intersection(ub.buffer(tol_m)).length / la,
            "b_em_a": ub.intersection(ua.buffer(tol_m)).length / lb}


# --------------------------------------------------------------------------------------
# gold validation sample (stratified by class x AOI) for manual interpretation in QGIS
# --------------------------------------------------------------------------------------
GOLD_COLUMNS = ["rotulo_manual", "confianca", "interprete", "obs"]


def stratified_sample(labels: dict[str, np.ndarray], grids: dict[str, Grid], n_per_class: int, cfg: dict,
                      seed: int) -> "pd.DataFrame":
    rng = np.random.default_rng(seed)
    names = cfg["classes"]["nomes"]
    rows = []
    for aoi, lab in labels.items():
        g = grids[aoi]
        for cid, cname in enumerate(names):
            r, c = np.nonzero(lab == cid)
            if len(r) == 0:
                continue
            pick = rng.choice(len(r), size=min(n_per_class, len(r)), replace=False)
            lon, lat = g.pixel_lonlat(r[pick], c[pick])
            for i, p in enumerate(pick):
                rows.append({"id": f"{aoi}_{cname}_{i:03d}", "aoi": aoi, "lon": lon[i], "lat": lat[i],
                             "linha": int(r[p]), "coluna": int(c[p]), "res_m": g.res, "classe_automatica": cname})
    df = pd.DataFrame(rows)
    for c in GOLD_COLUMNS:
        df[c] = ""
    return df


def export_gold(df: pd.DataFrame, cfg: dict) -> tuple[Path, bool]:
    """Write CSV + GPKG only if they do not exist yet (never overwrite manual labels)."""
    import geopandas as gpd

    csv = ROOT / cfg["validacao_ouro"]["arquivo_csv"]
    gpkg = ROOT / cfg["validacao_ouro"]["arquivo_gpkg"]
    if csv.exists():
        return csv, False
    csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False, encoding="utf-8")
    gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326).to_file(gpkg, driver="GPKG")
    return csv, True


def load_gold(cfg: dict) -> pd.DataFrame | None:
    """Return the gold points that have a manual label, or None."""
    csv = ROOT / cfg["validacao_ouro"]["arquivo_csv"]
    gpkg = ROOT / cfg["validacao_ouro"]["arquivo_gpkg"]
    df = None
    if gpkg.exists():  # QGIS edits usually land in the GPKG
        import geopandas as gpd

        df = pd.DataFrame(gpd.read_file(gpkg).drop(columns="geometry"))
    elif csv.exists():
        df = pd.read_csv(csv, dtype=str)
    if df is None or "rotulo_manual" not in df:
        return None
    df = df[df["rotulo_manual"].fillna("").astype(str).str.strip() != ""].copy()
    valid = set(cfg["classes"]["nomes"])
    df = df[df["rotulo_manual"].str.strip().isin(valid)]
    for c in ("linha", "coluna", "res_m"):
        df[c] = df[c].astype(int)
    return df if len(df) else None
