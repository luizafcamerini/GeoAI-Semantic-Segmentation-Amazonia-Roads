"""Configuration loading and AOI grid definitions."""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from affine import Affine
from pyproj import CRS, Transformer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "resultados"
VALID_DIR = ROOT / "validacao"

GRID_UNIT_M = 960  # lcm(10, 30, 64): nested grids at every study resolution


def load_config(path: str | Path = ROOT / "config.yaml", modo_rapido: bool | None = None) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if modo_rapido is None:
        env = os.environ.get("MODO_RAPIDO")
        modo_rapido = cfg["projeto"]["modo_rapido"] if env is None else env not in ("0", "false", "False")
    cfg["projeto"]["modo_rapido"] = bool(modo_rapido)
    cfg["run"] = cfg["modos"]["rapido" if modo_rapido else "completo"]
    names = cfg["classes"]["nomes"]
    cfg["classes"]["id"] = {n: i for i, n in enumerate(names)}
    for d in (DATA, RESULTS, VALID_DIR):
        d.mkdir(exist_ok=True)
    return cfg


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def aois_brasil(cfg: dict) -> list[str]:
    return [a for a in cfg["run"]["aois"] if cfg["aois"][a]["pais"] == "BR"]


def aois_por_papel(cfg: dict, papel: str) -> list[str]:
    return [a for a in cfg["run"]["aois"] if cfg["aois"][a]["papel"] == papel]


@dataclass(frozen=True)
class Grid:
    """A north-up raster grid in a metric (UTM) CRS."""

    crs: CRS
    minx: float
    maxy: float
    size_m: int
    res: int

    @property
    def shape(self) -> tuple[int, int]:
        n = self.size_m // self.res
        return n, n

    @property
    def transform(self) -> Affine:
        return Affine(self.res, 0, self.minx, 0, -self.res, self.maxy)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.minx, self.maxy - self.size_m, self.minx + self.size_m, self.maxy

    def at(self, res: int) -> "Grid":
        assert self.size_m % res == 0, (self.size_m, res)
        return Grid(self.crs, self.minx, self.maxy, self.size_m, res)

    def bounds_lonlat(self, pad_deg: float = 0.0) -> tuple[float, float, float, float]:
        t = Transformer.from_crs(self.crs, 4326, always_xy=True)
        x0, y0, x1, y1 = self.bounds
        xs, ys = t.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
        return min(xs) - pad_deg, min(ys) - pad_deg, max(xs) + pad_deg, max(ys) + pad_deg

    def pixel_lonlat(self, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = self.minx + (cols + 0.5) * self.res
        y = self.maxy - (rows + 0.5) * self.res
        t = Transformer.from_crs(self.crs, 4326, always_xy=True)
        return t.transform(x, y)


def utm_crs(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg((32700 if lat < 0 else 32600) + zone)


def aoi_grid(cfg: dict, aoi: str, res: int = 10) -> Grid:
    lon, lat = cfg["aois"][aoi]["centro"]
    size = int(cfg["run"]["tamanho_aoi_m"])
    assert size % GRID_UNIT_M == 0
    crs = utm_crs(lon, lat)
    cx, cy = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
    # snap the corner to the 960 m lattice so grids are reproducible
    minx = np.floor((cx - size / 2) / GRID_UNIT_M) * GRID_UNIT_M
    maxy = np.ceil((cy + size / 2) / GRID_UNIT_M) * GRID_UNIT_M
    return Grid(crs, float(minx), float(maxy), size, res)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def load_json(path: Path, default=None):
    if not Path(path).exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)
