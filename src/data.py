"""Data acquisition: STAC discovery, windowed COG reads onto AOI grids, local caching.

Every remote source records its outcome in data/fontes_status.json so the notebook can
report what was actually used and what failed. Nothing is fabricated when a source fails.
"""
from __future__ import annotations

import datetime as dt
import gzip
import io
import json
import math
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject

from .config import DATA, Grid, load_json, save_json

STATUS_FILE = DATA / "fontes_status.json"
CHOICES_FILE = DATA / "escolhas_colecoes.json"

GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff,.zip,.gpkg,.shp,.shx,.dbf,.prj",
    GDAL_HTTP_MAX_RETRY="4",
    GDAL_HTTP_RETRY_DELAY="2",
    GDAL_HTTP_TIMEOUT="60",
    VSI_CACHE="TRUE",
    GDAL_HTTP_MULTIRANGE="YES",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
)

# S2 L2A scene classification values considered clear: vegetation, bare, water, unclassified, snow
S2_SCL_CLEAR = (4, 5, 6, 7, 11)
# BDC CBERS/WFI CMASK: 127 = clear, 255 = cloud/shadow, 0 = no data (verified by histogram in the notebook)
WFI_CMASK_CLEAR = (127,)


# --------------------------------------------------------------------------------------
# status log
# --------------------------------------------------------------------------------------
def log_status(fonte: str, ok: bool, detalhe: str = "") -> None:
    st = load_json(STATUS_FILE, {})
    st[fonte] = {"ok": bool(ok), "detalhe": str(detalhe)[:500], "quando": dt.datetime.now().isoformat(timespec="seconds")}
    save_json(st, STATUS_FILE)


def status_table():
    import pandas as pd

    st = load_json(STATUS_FILE, {})
    return pd.DataFrame([{"fonte": k, **v} for k, v in st.items()]).sort_values("fonte") if st else pd.DataFrame()


def http_get(url: str, cfg: dict, **kw) -> requests.Response:
    headers = {"User-Agent": cfg["projeto"]["user_agent"]}
    r = requests.get(url, headers=headers, timeout=kw.pop("timeout", cfg["fontes"]["timeout_s"]), **kw)
    r.raise_for_status()
    return r


def download(url: str, dest: Path, cfg: dict) -> Path:
    """Download once into the cache; skip if the file already exists."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = http_get(url, cfg, stream=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with open(tmp, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    tmp.replace(dest)
    return dest


# --------------------------------------------------------------------------------------
# STAC discovery
# --------------------------------------------------------------------------------------
def list_collections(stac_url: str) -> list[dict]:
    from pystac_client import Client

    client = Client.open(stac_url)
    return [{"id": c.id, "title": c.title or ""} for c in client.get_collections()]


def choose_collection(collections: list[dict], padrao: dict, key: str) -> str | None:
    """Pick a collection whose id+title contains all `incluir` terms (case-insensitive).

    Among matches, the highest version suffix wins (e.g. S2-16D-2 over S2-16D-1).
    """
    def text(c):
        return f"{c['id']} {c['title']}".lower()

    hits = [c for c in collections
            if all(t in text(c) for t in padrao["incluir"]) and not any(t in text(c) for t in padrao.get("excluir", []))]
    if not hits:
        log_status(f"stac:{key}", False, f"nenhuma coleção casa {padrao}")
        return None

    def version(c):
        tail = c["id"].rsplit("-", 1)[-1]
        return int(tail) if tail.isdigit() else 0

    best = max(hits, key=version)
    choices = load_json(CHOICES_FILE, {})
    choices[key] = {"escolhida": best["id"], "titulo": best["title"], "candidatas": [h["id"] for h in hits]}
    save_json(choices, CHOICES_FILE)
    return best["id"]


def search_items(stac_url: str, collection: str, bbox_ll, year: int, sign: bool = False, query: dict | None = None) -> list:
    from pystac_client import Client

    modifier = None
    if sign:
        import planetary_computer

        modifier = planetary_computer.sign_inplace
    client = Client.open(stac_url, modifier=modifier)
    search = client.search(collections=[collection], bbox=list(bbox_ll),
                           datetime=f"{year}-01-01/{year}-12-31", query=query, max_items=5000)
    items = list(search.items())
    items.sort(key=lambda it: it.datetime or it.properties.get("start_datetime"))
    return items


def item_date(item) -> dt.date:
    d = item.datetime or dt.datetime.fromisoformat(item.properties["start_datetime"].replace("Z", "+00:00"))
    return d.date()


def group_by_date(items) -> dict[dt.date, list]:
    out: dict[dt.date, list] = {}
    for it in items:
        out.setdefault(item_date(it), []).append(it)
    return dict(sorted(out.items()))


def pick_regular(dates: list, n: int) -> list:
    if len(dates) <= n:
        return list(dates)
    idx = np.unique(np.round(np.linspace(0, len(dates) - 1, n)).astype(int))
    return [dates[i] for i in idx]


# --------------------------------------------------------------------------------------
# raster reads
# --------------------------------------------------------------------------------------
def read_to_grid(href: str, grid: Grid, resampling=Resampling.nearest, dtype="float32",
                 src_nodata=None, dst_nodata=np.nan, band: int = 1) -> np.ndarray:
    """Read the part of a (remote) raster covering `grid`, warped onto it."""
    path = "/vsicurl/" + href if href.startswith("http") else href
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(path) as src:
            nod = src.nodata if src_nodata is None else src_nodata
            with WarpedVRT(src, crs=grid.crs, transform=grid.transform, width=grid.shape[1], height=grid.shape[0],
                           resampling=resampling, src_nodata=nod, nodata=nod) as vrt:
                arr = vrt.read(band).astype(dtype)
                if nod is not None and dtype.startswith("float"):
                    arr[arr == nod] = dst_nodata
    return arr


def _retry(fn, tries=3, wait=3):
    for k in range(tries):
        try:
            return fn()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(wait * (k + 1))


def mosaic_first_valid(arrs: list[np.ndarray], nodata=np.nan) -> np.ndarray:
    out = arrs[0].copy()
    for a in arrs[1:]:
        fill = np.isnan(out) if (isinstance(nodata, float) and np.isnan(nodata)) else (out == nodata)
        out[fill] = a[fill]
    return out


# --------------------------------------------------------------------------------------
# data cubes (BDC): regular composites, one or more tiles per date
# --------------------------------------------------------------------------------------
SENSORS = {
    # bands are renamed to a common vocabulary: blue, green, red, nir, swir1, swir2, rededge
    "s2": {"bands": {"B02": "blue", "B03": "green", "B04": "red", "B05": "rededge", "B08": "nir",
                     "B11": "swir1", "B12": "swir2"},
           "mask": "SCL", "clear": S2_SCL_CLEAR, "res": 10, "scale": 1e-4},
    "cbers": {"bands": {"BAND13": "blue", "BAND14": "green", "BAND15": "red", "BAND16": "nir"},
              "mask": "CMASK", "clear": WFI_CMASK_CLEAR, "res": 64, "scale": 1e-4},
    "landsat": {"bands": {"blue": "blue", "green": "green", "red": "red", "nir08": "nir",
                          "swir16": "swir1", "swir22": "swir2"},
                "mask": "qa_pixel", "clear": None, "res": 30, "scale": 2.75e-5, "offset": -0.2},
}


def _landsat_clear(qa: np.ndarray) -> np.ndarray:
    q = np.nan_to_num(qa, nan=1).astype(np.uint16)
    bad = (q & (1 << 0)) | (q & (1 << 1)) | (q & (1 << 3)) | (q & (1 << 4))  # fill, dilated cloud, cloud, shadow
    return bad == 0


def cube_path(aoi: str, key: str) -> Path:
    return DATA / "cubos" / f"{aoi}__{key}.npz"


def load_bdc_cube(cfg: dict, aoi: str, grid: Grid, sensor: str, collection: str, max_dates: int,
                  workers: int = 8) -> dict:
    """Read a BDC regular cube onto `grid`. Returns dict(data int16 [T,C,H,W], valid bool [T,H,W], dates, bands).

    Reflectance is stored as int16 (x 10000) to keep the cache small; NaN/invalid -> -9999.
    """
    key = f"{sensor}_{grid.res}"
    path = cube_path(aoi, key)
    if path.exists():
        return dict(np.load(path, allow_pickle=True))
    spec = SENSORS[sensor]
    year = cfg["projeto"]["ano"]
    items = search_items(cfg["fontes"]["bdc_stac"], collection, grid.bounds_lonlat(0.01), year)
    if not items:
        raise RuntimeError(f"sem itens {collection} para {aoi}")
    by_date = group_by_date(items)
    dates = pick_regular(list(by_date), max_dates)
    asset_names = list(spec["bands"]) + [spec["mask"]]

    def read_one(args):
        d, a = args
        hrefs = [it.assets[a].href for it in by_date[d] if a in it.assets]
        rs = Resampling.nearest if a == spec["mask"] else Resampling.bilinear
        arrs = [_retry(lambda h=h: read_to_grid(h, grid, rs)) for h in hrefs]
        return (d, a), mosaic_first_valid(arrs)

    jobs = [(d, a) for d in dates for a in asset_names]
    with ThreadPoolExecutor(workers) as ex:
        res = dict(ex.map(read_one, jobs))

    T, C = len(dates), len(spec["bands"])
    H, W = grid.shape
    data = np.full((T, C, H, W), -9999, np.int16)
    valid = np.zeros((T, H, W), bool)
    for t, d in enumerate(dates):
        m = res[(d, spec["mask"])]
        ok = _landsat_clear(m) if sensor == "landsat" else np.isin(np.nan_to_num(m, nan=-1), spec["clear"])
        for c, b in enumerate(spec["bands"]):
            v = res[(d, b)] * spec["scale"] + spec.get("offset", 0.0)
            ok &= np.isfinite(v)
            data[t, c] = np.where(np.isfinite(v), np.clip(v * 1e4, -9998, 32767), -9999).astype(np.int16)
        valid[t] = ok
    out = dict(data=data, valid=valid, dates=np.array([str(d) for d in dates]),
               bands=np.array(list(spec["bands"].values())), res=grid.res, sensor=sensor, fonte=f"BDC:{collection}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


# --------------------------------------------------------------------------------------
# Sentinel-2 L2A from the Planetary Computer, composited to 16-day bins (Europe / fallback)
# --------------------------------------------------------------------------------------
def load_pc_s2_cube(cfg: dict, aoi: str, grid: Grid, max_dates: int, scenes_per_bin: int = 2,
                    max_cloud: float = 60, workers: int = 8) -> dict:
    """Median of clear observations per 16-day bin (same calendar as the BDC S2-16D cube)."""
    key = f"s2_{grid.res}"
    path = cube_path(aoi, key)
    if path.exists():
        return dict(np.load(path, allow_pickle=True))
    spec = SENSORS["s2"]
    year = cfg["projeto"]["ano"]
    items = search_items(cfg["fontes"]["pc_stac"], "sentinel-2-l2a", grid.bounds_lonlat(0.01), year, sign=True,
                         query={"eo:cloud_cover": {"lt": max_cloud}})
    if not items:
        raise RuntimeError(f"sem cenas S2 L2A no PC para {aoi}")
    start = dt.date(year, 1, 1)
    bins: dict[int, list] = {}
    for it in items:
        k = (item_date(it) - start).days // 16
        bins.setdefault(min(k, 22), []).append(it)
    all_bins = sorted(bins)
    chosen = pick_regular(all_bins, max_dates)
    asset_names = list(spec["bands"]) + ["SCL"]

    def read_scene(args):
        it, a = args
        rs = Resampling.nearest if a == "SCL" else Resampling.bilinear
        return (it.id, a), _retry(lambda: read_to_grid(it.assets[a].href, grid, rs, src_nodata=0))

    scenes = {k: sorted(bins[k], key=lambda it: it.properties.get("eo:cloud_cover", 100))[:scenes_per_bin] for k in chosen}
    jobs = [(it, a) for k in chosen for it in scenes[k] for a in asset_names]
    with ThreadPoolExecutor(workers) as ex:
        res = dict(ex.map(read_scene, jobs))

    T, C = len(chosen), len(spec["bands"])
    H, W = grid.shape
    data = np.full((T, C, H, W), -9999, np.int16)
    valid = np.zeros((T, H, W), bool)
    for t, k in enumerate(chosen):
        stack = []
        for it in scenes[k]:
            ok = np.isin(np.nan_to_num(res[(it.id, "SCL")], nan=-1), S2_SCL_CLEAR)
            # processing baseline >= 04.00 adds a +1000 DN offset to L2A reflectance
            off = 1000 if float(it.properties.get("s2:processing_baseline", "0") or 0) >= 4.0 else 0
            bands = np.stack([(res[(it.id, b)] - off) for b in spec["bands"]])
            bands[:, ~ok] = np.nan
            stack.append(bands)
        med = np.nanmedian(np.stack(stack), axis=0) if len(stack) > 1 else stack[0]
        ok = np.all(np.isfinite(med), axis=0)
        data[t] = np.where(np.isfinite(med), np.clip(med, -9998, 32767), -9999).astype(np.int16)
        valid[t] = ok
    out = dict(data=data, valid=valid, dates=np.array([str(start + dt.timedelta(days=16 * k)) for k in chosen]),
               bands=np.array(list(spec["bands"].values())), res=grid.res, sensor="s2",
               fonte="PlanetaryComputer:sentinel-2-l2a (mediana 16 dias)")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


# --------------------------------------------------------------------------------------
# resolution degradation (same dates, mean aggregation)
# --------------------------------------------------------------------------------------
def degrade_cube(cube: dict, src_grid: Grid, dst_res: int, min_valid_frac: float = 0.5) -> dict:
    """Aggregate a 10 m cube to `dst_res` by averaging valid sub-pixels (area-weighted)."""
    dst = src_grid.at(dst_res)
    T, C, H, W = cube["data"].shape
    h, w = dst.shape
    data = np.full((T, C, h, w), -9999, np.int16)
    valid = np.zeros((T, h, w), bool)
    for t in range(T):
        vf = np.zeros((h, w), np.float32)
        reproject(cube["valid"][t].astype(np.float32), vf, src_transform=src_grid.transform, src_crs=src_grid.crs,
                  dst_transform=dst.transform, dst_crs=dst.crs, resampling=Resampling.average)
        ok = vf >= min_valid_frac
        for c in range(C):
            src = cube["data"][t, c].astype(np.float32)
            src[~cube["valid"][t]] = np.nan
            out = np.full((h, w), np.nan, np.float32)
            reproject(src, out, src_transform=src_grid.transform, src_crs=src_grid.crs, src_nodata=np.nan,
                      dst_transform=dst.transform, dst_crs=dst.crs, dst_nodata=np.nan, resampling=Resampling.average)
            good = ok & np.isfinite(out)
            data[t, c] = np.where(good, out, -9999).astype(np.int16)
        valid[t] = ok & np.all(data[t] != -9999, axis=0)
    return dict(data=data, valid=valid, dates=cube["dates"], bands=cube["bands"], res=dst_res,
                sensor=str(cube["sensor"]), fonte=f"{cube['fonte']} degradado {src_grid.res}->{dst_res} m (média)")


# --------------------------------------------------------------------------------------
# MapBiomas (latest collection discovered from the public bucket)
# --------------------------------------------------------------------------------------
def mapbiomas_latest(cfg: dict, year: int) -> tuple[str, int, str]:
    """Return (collection_name, year_used, https_url) discovered by listing the public bucket."""
    bucket, prefix = cfg["fontes"]["mapbiomas_bucket"], cfg["fontes"]["mapbiomas_prefixo"]
    api = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o"
    r = http_get(api, cfg, params={"prefix": prefix + "collection_", "delimiter": "/", "fields": "prefixes"})
    cols = [p.rstrip("/").split("/")[-1] for p in r.json().get("prefixes", [])]
    cols.sort(key=lambda c: int("".join(ch for ch in c.split("_")[-1] if ch.isdigit()) or 0), reverse=True)
    for col in cols:
        r = http_get(api, cfg, params={"prefix": f"{prefix}{col}/lulc/coverage/brazil_coverage_", "fields": "items(name)"})
        names = [i["name"] for i in r.json().get("items", [])]
        years = sorted(int(n[-8:-4]) for n in names if n.endswith(".tif") and n[-8:-4].isdigit())
        if years:
            y = year if year in years else max(y for y in years if y <= year) if min(years) <= year else years[-1]
            url = f"https://storage.googleapis.com/{bucket}/{prefix}{col}/lulc/coverage/brazil_coverage_{y}.tif"
            choices = load_json(CHOICES_FILE, {})
            choices["mapbiomas"] = {"colecao": col, "ano": y, "url": url, "colecoes_no_bucket": cols}
            save_json(choices, CHOICES_FILE)
            return col, y, url
    raise RuntimeError("nenhuma coleção MapBiomas com GeoTIFF no bucket")


def load_mapbiomas(cfg: dict, aoi: str, grid: Grid) -> np.ndarray:
    path = DATA / "rotulos" / f"{aoi}__mapbiomas_{grid.res}.npy"
    if path.exists():
        return np.load(path)
    col, y, url = mapbiomas_latest(cfg, cfg["projeto"]["ano"])
    rs = Resampling.nearest if grid.res <= 30 else Resampling.mode
    arr = _retry(lambda: read_to_grid(url, grid, rs, dtype="uint8", src_nodata=0))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    log_status("MapBiomas", True, f"{col}, ano {y}")
    return arr


# --------------------------------------------------------------------------------------
# Planetary Computer rasters: ESA WorldCover, JRC Global Surface Water
# --------------------------------------------------------------------------------------
def load_pc_raster(cfg: dict, aoi: str, grid: Grid, collection: str, asset: str, year: int | None = None,
                   resampling=Resampling.nearest) -> np.ndarray:
    path = DATA / "rotulos" / f"{aoi}__{collection}_{asset}_{grid.res}.npy"
    if path.exists():
        return np.load(path)
    import planetary_computer
    from pystac_client import Client

    client = Client.open(cfg["fontes"]["pc_stac"], modifier=planetary_computer.sign_inplace)
    items = list(client.search(collections=[collection], bbox=list(grid.bounds_lonlat(0.01))).items())
    if not items:
        raise RuntimeError(f"{collection}: sem itens")
    if year is not None:
        items.sort(key=lambda it: abs(item_date(it).year - year))
        y0 = item_date(items[0]).year
        items = [it for it in items if item_date(it).year == y0]
    arrs = [_retry(lambda it=it: read_to_grid(it.assets[asset].href, grid, resampling, src_nodata=None))
            for it in items]
    arr = mosaic_first_valid(arrs)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    log_status(f"PC:{collection}", True, f"{len(items)} itens; {item_date(items[0])}")
    return arr


# --------------------------------------------------------------------------------------
# vector sources
# --------------------------------------------------------------------------------------
def load_osm_roads(cfg: dict, aoi: str, grid: Grid):
    """OSM highways inside the AOI via osmnx (Overpass). Cached as GeoPackage."""
    import geopandas as gpd

    path = DATA / "vetores" / f"{aoi}__osm_estradas.gpkg"
    if path.exists():
        return gpd.read_file(path)
    import osmnx as ox

    ox.settings.http_user_agent = cfg["projeto"]["user_agent"]
    ox.settings.requests_timeout = 180
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(DATA / "osmnx_cache")
    w, s, e, n = grid.bounds_lonlat(0.002)
    # public Overpass instances are often at capacity: retry with backoff and mirrors
    endpoints = [cfg["fontes"]["overpass"]] + cfg["fontes"].get("overpass_espelhos", [])
    erros = []
    gdf = None
    for tentativa in range(3):
        for url in endpoints:
            ox.settings.overpass_url = url.rsplit("/interpreter", 1)[0]
            try:
                gdf = ox.features.features_from_bbox((w, s, e, n), tags={"highway": True})
                break
            except Exception as ex:  # noqa: BLE001
                erros.append(f"{url}: {str(ex)[:80]}")
        if gdf is not None:
            break
        time.sleep(20 * (tentativa + 1))
    if gdf is None:
        raise RuntimeError("Overpass indisponível: " + " | ".join(erros[-3:]))
    gdf = gdf[gdf.geom_type.isin(["LineString", "MultiLineString"])].reset_index()
    gdf = gdf[~gdf["highway"].astype(str).isin(cfg["estradas"]["tipos_excluidos"])]
    keep = [c for c in ["osmid", "highway", "surface", "name", "geometry"] if c in gdf.columns]
    gdf = gdf[keep].copy()
    for c in keep:
        if c != "geometry":
            gdf[c] = gdf[c].astype(str)
    gdf = gdf.to_crs(grid.crs)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GPKG")
    log_status(f"OSM:{aoi}", True, f"{len(gdf)} vias")
    return gdf


def _bc250_zip_url(cfg: dict) -> str:
    import re

    base = cfg["fontes"]["bc250_indice"]
    vers = sorted(set(re.findall(r'href="(versao\d{4})/"', http_get(base, cfg).text)))
    for v in reversed(vers):
        for sub in ("shapefiles/", "shapefile/"):
            try:
                html = http_get(base + v + "/" + sub, cfg).text
            except Exception:
                continue
            z = re.findall(r'href="([^"]+\.zip)"', html)
            if z:
                return base + v + "/" + sub + z[0]
    raise RuntimeError("BC250: nenhum zip de shapefiles encontrado")


def load_bc250_roads(cfg: dict, aoi: str, grid: Grid, timeout_s: int | None = None):
    """IBGE BC250 road segments, read straight from the remote zip (no full 800 MB download).

    GDAL /vsizip//vsicurl/ fetches only the needed byte ranges; still slow, so it runs with a timeout.
    """
    import geopandas as gpd
    import pyogrio

    path = DATA / "vetores" / f"{aoi}__bc250_estradas.gpkg"
    if path.exists():
        return gpd.read_file(path)
    url = _bc250_zip_url(cfg)
    root = f"/vsizip//vsicurl/{url}"
    members = _zip_members(url, cfg)
    shp = [m for m in members if m.lower().endswith(".shp") and "rodovi" in m.lower()]
    if not shp:
        raise RuntimeError(f"BC250: camada rodoviária não encontrada no zip ({len(members)} membros)")
    w, s, e, n = grid.bounds_lonlat(0.002)
    frames = []
    with rasterio.Env(**GDAL_ENV):
        for m in shp:
            frames.append(pyogrio.read_dataframe(f"{root}/{m}", bbox=(w, s, e, n)))
    import pandas as pd

    gdf = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs).to_crs(grid.crs)
    gdf = gdf[gdf.geom_type.isin(["LineString", "MultiLineString"])]
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf[["geometry"]].assign(fonte="BC250").to_file(path, driver="GPKG")
    log_status(f"BC250:{aoi}", True, f"{len(gdf)} trechos; {url.rsplit('/', 1)[-1]}")
    return gdf


def _zip_members(url: str, cfg: dict) -> list[str]:
    """List members of a remote zip by reading only its central directory (HTTP range requests)."""
    head = requests.head(url, headers={"User-Agent": cfg["projeto"]["user_agent"]}, timeout=60, allow_redirects=True)
    size = int(head.headers["Content-Length"])
    tail = requests.get(url, headers={"User-Agent": cfg["projeto"]["user_agent"],
                                      "Range": f"bytes={size - 65536}-{size - 1}"}, timeout=120).content
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:  # zip64
        i64 = tail.rfind(b"PK\x06\x06")
        cd_size = int.from_bytes(tail[i64 + 40:i64 + 48], "little")
        cd_off = int.from_bytes(tail[i64 + 48:i64 + 56], "little")
    else:
        cd_size = int.from_bytes(tail[i + 12:i + 16], "little")
        cd_off = int.from_bytes(tail[i + 16:i + 20], "little")
        if cd_off == 0xFFFFFFFF:
            i64 = tail.rfind(b"PK\x06\x06")
            cd_size = int.from_bytes(tail[i64 + 40:i64 + 48], "little")
            cd_off = int.from_bytes(tail[i64 + 48:i64 + 56], "little")
    cd = requests.get(url, headers={"User-Agent": cfg["projeto"]["user_agent"],
                                    "Range": f"bytes={cd_off}-{cd_off + cd_size - 1}"}, timeout=120).content
    names, p = [], 0
    while p + 46 <= len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        n = int.from_bytes(cd[p + 28:p + 30], "little")
        x = int.from_bytes(cd[p + 30:p + 32], "little")
        c = int.from_bytes(cd[p + 32:p + 34], "little")
        names.append(cd[p + 46:p + 46 + n].decode("utf-8", "replace"))
        p += 46 + n + x + c
    return names


def load_dnit_snv(cfg: dict, aoi: str, grid: Grid):
    """DNIT SNV via WFS (federal highways). Raises if the service is unreachable."""
    import geopandas as gpd

    path = DATA / "vetores" / f"{aoi}__dnit_snv.gpkg"
    if path.exists():
        return gpd.read_file(path)
    base = cfg["fontes"]["dnit_snv_wfs"]
    caps = http_get(base, cfg, params={"service": "WFS", "request": "GetCapabilities"}, timeout=30).text
    import re

    layers = [n for n in re.findall(r"<Name>([^<]+)</Name>", caps) if "snv" in n.lower()]
    layers.sort(key=lambda n: (not n.lower().endswith("snv_rod"), "teste" in n.lower()))  # road layer first
    if not layers:
        raise RuntimeError("DNIT WFS sem camada SNV")
    w, s, e, n = grid.bounds_lonlat(0.002)
    r = http_get(base, cfg, params={"service": "WFS", "version": "2.0.0", "request": "GetFeature",
                                    "typeNames": layers[0], "outputFormat": "application/json",
                                    "bbox": f"{s},{w},{n},{e},urn:ogc:def:crs:EPSG::4326"})
    gdf = gpd.read_file(io.BytesIO(r.content)).to_crs(grid.crs)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf[["geometry"]].assign(fonte="DNIT").to_file(path, driver="GPKG")
    log_status(f"DNIT:{aoi}", True, f"{len(gdf)} trechos ({layers[0]})")
    return gdf


def _quadkeys(bbox_ll, zoom: int) -> list[str]:
    def tile(lon, lat):
        lat = max(min(lat, 85.0511), -85.0511)
        n = 2 ** zoom
        x = int((lon + 180) / 360 * n)
        y = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
        return x, y

    w, s, e, n = bbox_ll
    x0, y1 = tile(w, s)
    x1, y0 = tile(e, n)
    keys = []
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            q = ""
            for z in range(zoom, 0, -1):
                d = 0
                m = 1 << (z - 1)
                d += 1 if x & m else 0
                d += 2 if y & m else 0
                q += str(d)
            keys.append(q)
    return keys


def load_ms_buildings(cfg: dict, aoi: str, grid: Grid, country: str = "Brazil"):
    """Microsoft Global ML Building Footprints for the AOI (quadkey zoom-9 files)."""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import shape

    path = DATA / "vetores" / f"{aoi}__ms_buildings.gpkg"
    if path.exists():
        return gpd.read_file(path)
    links_path = download(cfg["fontes"]["ms_buildings_links"], DATA / "externos" / "ms_buildings_links.csv", cfg)
    links = pd.read_csv(links_path, dtype={"QuadKey": str})
    links = links[links["Location"] == country]
    bbox = grid.bounds_lonlat(0.002)
    wanted = set(_quadkeys(bbox, 9))
    rows = links[links["QuadKey"].isin(wanted)]
    if rows.empty:
        raise RuntimeError(f"MS Buildings: sem arquivos para {country} na AOI")
    geoms = []
    for url in rows["Url"]:
        raw = gzip.decompress(http_get(url, cfg, timeout=300).content)
        for line in raw.splitlines():
            g = shape(json.loads(line)["geometry"])
            geoms.append(g)
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=4326)
    w, s, e, n = bbox
    gdf = gdf.cx[w:e, s:n].to_crs(grid.crs)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GPKG")
    log_status(f"MSBuildings:{aoi}", True, f"{len(gdf)} edificações")
    return gdf


def load_buildings(cfg: dict, aoi: str, grid: Grid):
    """Building footprints: Microsoft (if the AOI has a file) else OSM `building=*`. Returns (gdf, source)."""
    try:
        return load_ms_buildings(cfg, aoi, grid, "Brazil" if cfg["aois"][aoi]["pais"] == "BR" else "Portugal"), "Microsoft"
    except Exception as e:  # noqa: BLE001
        log_status(f"MSBuildings:{aoi}", False, e)
    import geopandas as gpd

    path = DATA / "vetores" / f"{aoi}__osm_edificacoes.gpkg"
    if path.exists():
        return gpd.read_file(path), "OSM"
    import osmnx as ox

    ox.settings.http_user_agent = cfg["projeto"]["user_agent"]
    ox.settings.cache_folder = str(DATA / "osmnx_cache")
    try:
        gdf = ox.features.features_from_bbox(grid.bounds_lonlat(0.002), tags={"building": True})
        gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])][["geometry"]].reset_index(drop=True)
    except Exception as e:  # noqa: BLE001  (osmnx raises when the bbox has no building)
        log_status(f"OSMEdificacoes:{aoi}", False, e)
        gdf = gpd.GeoDataFrame(geometry=[], crs=4326)
    gdf = gdf.to_crs(grid.crs)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GPKG")
    log_status(f"OSMEdificacoes:{aoi}", True, f"{len(gdf)} edificações")
    return gdf, "OSM"


def load_mapbiomas_validation_points(cfg: dict):
    """Official MapBiomas accuracy-assessment points (LAPIG/TVI). Returns GeoDataFrame or raises."""
    import geopandas as gpd

    z = download(cfg["fontes"]["mapbiomas_pontos_validacao"], DATA / "externos" / "mapbiomas_pontos_validacao.zip", cfg)
    with zipfile.ZipFile(z) as zf:
        members = zf.namelist()
    vec = [m for m in members if m.lower().endswith((".shp", ".gpkg", ".geojson", ".csv"))]
    if not vec:
        raise RuntimeError(f"zip sem camada vetorial: {members[:10]}")
    gdf = gpd.read_file(f"zip://{z}!{vec[0]}")
    # the LON/LAT attribute columns are swapped in the published file; trust the geometry instead
    gdf["lon"], gdf["lat"] = gdf.geometry.x, gdf.geometry.y
    years = sorted(int(c[6:]) for c in gdf.columns if c.startswith("CLASS_") and c[6:].isdigit())
    log_status("MapBiomas:pontos_validacao", True, f"{len(gdf)} pontos; {vec[0]}; anos {years[0]}-{years[-1]}")
    return gdf
