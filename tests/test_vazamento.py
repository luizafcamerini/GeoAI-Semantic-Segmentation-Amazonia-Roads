"""Leakage tests: no test pixel/block in training, holdout never trained on, metrics only from Brazil.

Synthetic checks always run; checks on the real saved splits / metrics run when the notebook has
been executed (resultados/splits/*.npz, resultados/metricas.csv).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import maximum_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import validation as V  # noqa: E402
from src.config import load_config  # noqa: E402

CFG = load_config(modo_rapido=True)
SPLITS = sorted((ROOT / "resultados" / "splits").glob("*.npz"))


@pytest.mark.parametrize("res", [10, 30, 64])
@pytest.mark.parametrize("fold", [0, 1, 2])
def test_blocos_sem_sobreposicao_e_com_margem(res, fold):
    size, bloco, buf = 9600, 1920, 640
    shape = (size // res, size // res)
    folds = V.assign_block_folds(size, bloco, 3, seed=42)
    tr, te = V.fold_masks(shape, res, bloco, buf, folds, fold)
    assert te.any() and tr.any()
    assert not (tr & te).any(), "pixel em treino e teste"
    b = math.ceil(buf / res)
    near = maximum_filter(te, size=2 * b + 1)
    assert not (tr & near).any(), "pixel de treino dentro da margem do teste"


def test_mesma_particao_em_todas_as_resolucoes():
    """A block is test at 10 m iff it is test at 64 m (blocks are defined in metres)."""
    size, bloco = 9600, 1920
    folds = V.assign_block_folds(size, bloco, 3, seed=42)
    _, te10 = V.fold_masks((960, 960), 10, bloco, 640, folds, 1)
    _, te64 = V.fold_masks((150, 150), 64, bloco, 640, folds, 1)
    # sample 64 m pixel centres and look them up at 10 m
    r, c = np.mgrid[0:150, 0:150]
    r10, c10 = ((r * 64 + 32) // 10), ((c * 64 + 32) // 10)
    agree = te10[r10, c10] == te64
    # only pixels straddling a block edge may differ
    assert agree.mean() > 0.95


def test_desenho_holdout_e_teste_no_brasil():
    exps = V.design_experiments(CFG)
    holdout = {a for a in CFG["aois"] if CFG["aois"][a]["papel"] == "holdout"}
    assert holdout, "nenhuma AOI de holdout configurada"
    for e in exps:
        assert not holdout & set(e["treino"]), e["nome"]
        assert all(CFG["aois"][a]["pais"] == "BR" for a in e["teste"]), e["nome"]
        if e["tipo"] in ("holdout", "loro", "transferencia"):
            assert not set(e["treino"]) & set(e["teste"]), e["nome"]
    tipos = {e["tipo"] for e in exps}
    assert {"blocos", "holdout", "loro"} <= tipos


def test_check_design_rejeita_vazamento():
    ruim = [{"tipo": "holdout", "nome": "x", "treino": ["amz_ro_holdout"], "teste": ["amz_br163"]}]
    with pytest.raises(AssertionError):
        V.check_design(ruim, CFG)
    ruim = [{"tipo": "loro", "nome": "y", "treino": ["amz_br163"], "teste": ["eu_alentejo"]}]
    with pytest.raises(AssertionError):
        V.check_design(ruim, CFG)


@pytest.mark.skipif(not SPLITS, reason="notebook ainda não executado (sem resultados/splits)")
@pytest.mark.parametrize("path", SPLITS, ids=lambda p: p.stem)
def test_splits_reais_sem_vazamento(path):
    z = np.load(path)
    res = int(z["res"])
    treino = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith("treino__")}
    teste = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith("teste__")}
    holdout = {a for a in CFG["aois"] if CFG["aois"][a]["papel"] == "holdout"}
    assert not holdout & set(treino), f"holdout no treino: {path.name}"
    assert all(CFG["aois"][a]["pais"] == "BR" for a in teste), f"teste fora do Brasil: {path.name}"
    n = int(z["size_m"]) // res
    buffer_m = int(z["buffer_m"])
    for a, tr in treino.items():
        if a not in teste:
            continue  # different AOIs: geographically separated by construction
        te = np.zeros(n * n, bool)
        te[teste[a]] = True
        assert not te[tr].any(), f"pixel de teste usado no treino ({a}, {path.name})"
        # buffer: no training pixel within buffer_m of a test pixel
        b = math.ceil(buffer_m / res)
        near = maximum_filter(te.reshape(n, n), size=2 * b + 1).ravel()
        assert not near[tr].any(), f"pixel de treino na margem do teste ({a}, {path.name})"


@pytest.mark.skipif(not (ROOT / "resultados" / "metricas.csv").exists(), reason="sem resultados/metricas.csv")
def test_metricas_somente_brasil():
    m = pd.read_csv(ROOT / "resultados" / "metricas.csv")
    assert len(m) > 0
    assert set(m["pais"]) == {"BR"}
    assert set(m["aoi"]) <= {a for a in CFG["aois"] if CFG["aois"][a]["pais"] == "BR"}
