"""Etapas do notebook em funções curtas: cada célula do notebook chama uma etapa da história."""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from . import data as D
from . import labels as L
from . import metrics as MT
from . import plots as PL
from . import validation as V
from .config import DATA, RESULTS, aoi_grid, aois_brasil
from .experiments import DATA_KEYS, Store, active_keys, evaluate_predictions, run_experiment

IGNORE = 255
DADOS_FIGURA = ["s2_10", "s2_30", "s2_64", "cbers_64"]


# ------------------------------------------------------------------ configuração
def ficha(cfg: dict) -> pd.DataFrame:
    """Os parâmetros que importam para entender o estudo (os valores vêm do config.yaml)."""
    r = cfg["run"]
    lado = r["tamanho_aoi_m"] / 1000
    return pd.Series({
        "modo": "rápido" if cfg["projeto"]["modo_rapido"] else "completo",
        "ano das imagens e dos rótulos": cfg["projeto"]["ano"],
        "classes": ", ".join(cfg["classes"]["nomes"]),
        "tamanho de cada área": f"{lado:.1f} km × {lado:.1f} km",
        "datas por série temporal": r["max_datas"],
        "resoluções comparadas": "10 m, 30 m e 64 m",
        "largura típica de uma estrada de terra": f"{cfg['estradas']['largura_referencia_m']} m",
        "pixels de treino por área": r["pixels_treino_por_aoi"],
        "validação em blocos": f"blocos de {r['bloco_m']} m, margem de {r['buffer_m']} m, {r['folds_blocos']} divisões",
        "modelos": ", ".join(sorted({m for ms in r["experimentos"].values() for m in ms})),
    }, name="valor").to_frame()


def areas(cfg: dict) -> pd.DataFrame:
    papel = {"treino": "aprender", "holdout": "prova final", "transferencia": "aprender (teste Europa → Brasil)"}
    return pd.DataFrame([{"área": a, "local": cfg["aois"][a]["nome"], "bioma": cfg["aois"][a]["bioma"],
                          "papel": papel[cfg["aois"][a]["papel"]]} for a in cfg["run"]["aois"]]).set_index("área")


def mapa_areas(cfg: dict):
    import folium

    cor = {"treino": "blue", "holdout": "red", "transferencia": "purple"}
    m = folium.Map(location=[-12, -50], zoom_start=4)
    for a in cfg["run"]["aois"]:
        lon, lat = cfg["aois"][a]["centro"]
        folium.Marker([lat, lon], tooltip=cfg["aois"][a]["nome"], icon=folium.Icon(color=cor[cfg["aois"][a]["papel"]])).add_to(m)
    return m


# ------------------------------------------------------------------ imagens
def carregar_imagens(cfg: dict):
    """Baixa (ou lê do cache em data/) as séries temporais de cada área em cada resolução."""
    store = Store(cfg)
    colecoes = D.list_collections(cfg["fontes"]["bdc_stac"])
    for chave, sensor in [("s2_cubo", "s2"), ("cbers_wfi_cubo", "cbers"), ("landsat_cubo", "landsat")]:
        store.collections[sensor] = D.choose_collection(colecoes, cfg["fontes"]["padroes_colecao"][chave], chave)
    chaves, br, disp = active_keys(cfg), set(aois_brasil(cfg)), {}
    for a in cfg["run"]["aois"]:
        for k in chaves:
            if DATA_KEYS[k][0] == "s2" or a in br:          # CBERS e Landsat do BDC só cobrem o Brasil
                disp[(a, k)] = store.available(a, k)
    return store, chaves, disp


def tabela_nuvens(store: Store, disp: dict) -> pd.DataFrame:
    nome = {"s2_10": "Sentinel-2", "cbers_64": "CBERS"}
    linhas = [{"área": a, "sensor": nome[k], "% de observações sem nuvem": 100 * store.cube(a, k)["valid"].mean()}
              for (a, k), ok in disp.items() if ok and k in nome]
    return pd.DataFrame(linhas).pivot(index="área", columns="sensor", values="% de observações sem nuvem").round(0)


def cor_verdadeira(c: dict) -> np.ndarray:
    """Mediana do ano em cores naturais, ignorando pixels com nuvem."""
    b = [list(map(str, c["bands"])).index(x) for x in ("red", "green", "blue")]
    d = np.where(c["valid"][:, None], c["data"][:, b].astype(float), np.nan) / 1e4
    img = np.nanmedian(d, 0).transpose(1, 2, 0)
    return np.nan_to_num(np.clip(img / np.nanpercentile(img, 98), 0, 1))


def mostrar_resolucoes(store: Store, aoi: str, disp: dict) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, k in zip(axes, DADOS_FIGURA):
        if disp.get((aoi, k)):
            ax.imshow(cor_verdadeira(store.cube(aoi, k)), interpolation="nearest")
        nome = {"s2": "Sentinel-2", "cbers": "CBERS real"}[DATA_KEYS[k][0]]
        ax.set_title(f"{nome} · {DATA_KEYS[k][1]} m", fontsize=10)
        ax.set_axis_off()
    plt.show()


# ------------------------------------------------------------------ rótulos
def montar_rotulos(store: Store, chaves: list[str]) -> list[int]:
    """MapBiomas (via Google Earth Engine) + estradas do OpenStreetMap, em cada resolução."""
    cfg = store.cfg
    res = sorted({DATA_KEYS[k][1] for k in chaves})
    if any(not (DATA / "rotulos" / f"{a}__mapbiomas_{r}.npy").exists() for a in aois_brasil(cfg) for r in res):
        D.ee_init(cfg)
    for a in cfg["run"]["aois"]:
        for r in res:
            store.labels(a, r)
    return res


def tabela_estradas(store: Store, res: list[int]) -> pd.DataFrame:
    cfg, e = store.cfg, store.cfg["classes"]["id"]["estrada"]
    linhas = []
    for r in res:
        pct = [100 * (store.labels(a, r) == e).sum() / (store.labels(a, r) != IGNORE).sum() for a in aois_brasil(cfg)]
        linhas.append({"pixel": f"{r} m", "o pixel é estrada se ela cobrir pelo menos": f"{100 * L.road_threshold(cfg, r):.0f} %",
                       "% da área que vira estrada": round(float(np.mean(pct)), 1)})
    return pd.DataFrame(linhas).set_index("pixel")


def mostrar_rotulos(store: Store, aoi: str, res: list[int]) -> None:
    nomes = store.cfg["classes"]["nomes"]
    fig, axes = plt.subplots(1, len(res) + 1, figsize=(3.6 * (len(res) + 1), 4))
    axes[0].imshow(cor_verdadeira(store.cube(aoi, "s2_10")))
    axes[0].set_title("imagem (10 m)", fontsize=10)
    axes[0].set_axis_off()
    for ax, r in zip(axes[1:], res):
        PL.show_classes(ax, store.labels(aoi, r), nomes, f"rótulo a {r} m")
    PL.class_legend(fig, nomes)
    plt.subplots_adjust(bottom=0.1)
    plt.show()


def exportar_pontos_ouro(store: Store) -> None:
    cfg, br = store.cfg, aois_brasil(store.cfg)
    n = cfg["validacao_ouro"]["pontos_por_classe_aoi_rapido" if cfg["projeto"]["modo_rapido"] else "pontos_por_classe_aoi_completo"]
    amostra = L.stratified_sample({a: store.labels(a, 10) for a in br}, {a: aoi_grid(cfg, a, 10) for a in br},
                                  n, cfg, cfg["projeto"]["seed"])
    arq, novo = L.export_gold(amostra, cfg)
    ouro = L.load_gold(cfg)
    print(f"{len(amostra)} pontos {'gravados' if novo else 'já existiam'} em {arq.name}; "
          f"conferidos à mão até agora: {0 if ouro is None else len(ouro)}")


# ------------------------------------------------------------------ validação
def mostrar_blocos(cfg: dict, aoi: str, res: list[int]) -> None:
    """Desenha a divisão treino/teste e confere, em todas as divisões, que nada vaza."""
    r_ = cfg["run"]
    folds = V.assign_block_folds(r_["tamanho_aoi_m"], r_["bloco_m"], r_["folds_blocos"], cfg["projeto"]["seed"])
    fig, axes = plt.subplots(1, len(res), figsize=(4.3 * len(res), 4.2))
    for ax, r in zip(axes, res):
        shape = aoi_grid(cfg, aoi, r).shape
        for f in range(r_["folds_blocos"]):
            tr, te = V.fold_masks(shape, r, r_["bloco_m"], r_["buffer_m"], folds, f)
            rep = V.leakage_report(tr, te, r, r_["buffer_m"])
            assert rep["pixels_compartilhados"] == 0 and rep["treino_dentro_do_buffer"] == 0, rep
            if f == 0:
                vis = np.zeros(shape)
                vis[tr], vis[te] = 1, 2
                ax.imshow(vis, cmap=ListedColormap(["#f2c14e", "#4c78a8", "#e45756"]), vmin=0, vmax=2, interpolation="nearest")
                ax.set_title(f"pixel de {r} m", fontsize=10)
                ax.set_axis_off()
    fig.legend(handles=[Patch(color="#4c78a8", label="treino"), Patch(color="#e45756", label="teste"),
                        Patch(color="#f2c14e", label="margem (fica de fora)")], loc="lower center", ncol=3, frameon=False)
    plt.show()
    print("Conferido: nenhum pixel de teste aparece no treino, em nenhuma resolução.")


def tabela_testes(exps: list[dict]) -> pd.DataFrame:
    vistos, linhas = set(), []
    for e in exps:
        if e["tipo"] == "blocos":
            if "blocos" in vistos:
                continue
            treino, teste = "parte de cada área de treino", "outros blocos da mesma área"
        else:
            treino, teste = ", ".join(e["treino"]), ", ".join(e["teste"])
        vistos.add(e["tipo"])
        linhas.append({"teste": e["tipo"] if e["tipo"] == "blocos" else e["nome"], "aprende com": treino, "é avaliado em": teste})
    return pd.DataFrame(linhas).set_index("teste")


# ------------------------------------------------------------------ treino e avaliação
def treinar_todos(store: Store, exps: list[dict], chaves: list[str], disp: dict) -> dict:
    pred, n, t0 = {}, 0, time.time()
    for e in exps:
        for m in store.cfg["run"]["experimentos"][e["tipo"]]:
            for k in chaves:
                if all(disp.get((a, k)) for a in e["treino"] + e["teste"]):
                    run_experiment(store, e, m, k, pred.setdefault((e["tipo"], m, k), {}), log=lambda msg: None)
                    n += 1
    print(f"{n} modelos treinados em {(time.time() - t0) / 60:.1f} min")
    return pred


def avaliar_todos(store: Store, pred: dict) -> pd.DataFrame:
    linhas = []
    for (tipo, m, k), mapas in pred.items():
        linhas += evaluate_predictions(store, mapas, tipo, m, k)
    met = pd.DataFrame(linhas).drop(columns="_cm")
    assert set(met.pais) == {"BR"}, "há métrica fora do Brasil!"
    met.to_csv(RESULTS / "metricas.csv", index=False)
    return met


def comparar_europa(met: pd.DataFrame, chaves: list[str]):
    if "transferencia" not in met.experimento.unique():
        print("teste Europa → Brasil não executado")
        return None
    comp = met[met.experimento == "transferencia"].merge(met[met.experimento.isin(["holdout", "loro"])],
                                                         on=["modelo", "dados", "aoi"], suffixes=("_EU", "_BR"))
    ordem = [k for k in chaves if k in comp.dados.unique()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (col, titulo) in zip(axes, [("f1_macro", "F1 médio das 6 classes"), ("completude", "estradas encontradas")]):
        comp.groupby("dados")[[f"{col}_BR", f"{col}_EU"]].mean().reindex(ordem).plot.bar(ax=ax, color=["#4c78a8", "#e45756"], rot=0)
        ax.set_title(titulo, fontsize=10)
        ax.set_xlabel("")
        ax.legend(["aprendeu no Brasil", "aprendeu na Europa"], frameon=False)
        ax.grid(alpha=.3, axis="y")
    plt.tight_layout()
    plt.show()
    return comp


def mostrar_erros(store: Store, pred: dict, modelo: str, aoi: str, chaves: list[str]) -> None:
    nomes = store.cfg["classes"]["nomes"]
    ks = [k for k in chaves if aoi in pred.get(("holdout", modelo, k), {})]
    fig, axes = plt.subplots(2, len(ks) + 1, figsize=(3 * (len(ks) + 1), 6), squeeze=False)
    PL.show_classes(axes[0, 0], store.labels(aoi, 10), nomes, "rótulo (10 m)")
    axes[1, 0].imshow(cor_verdadeira(store.cube(aoi, "s2_10")))
    axes[1, 0].set_title("imagem", fontsize=9)
    axes[1, 0].set_axis_off()
    for j, k in enumerate(ks, 1):
        p = pred[("holdout", modelo, k)][aoi]
        PL.show_classes(axes[0, j], p, nomes, f"previsto · {k}")
        err = PL.error_map(store.labels(aoi, DATA_KEYS[k][1]), p)
        axes[1, j].imshow(np.ma.masked_equal(err, IGNORE), cmap=ListedColormap(["#e8e8e8", "#d62728"]),
                          vmin=0, vmax=1, interpolation="nearest")
        axes[1, j].set_title(f"erros: {100 * (err == 1).sum() / max((err != IGNORE).sum(), 1):.0f} %", fontsize=9)
        axes[1, j].set_axis_off()
    PL.class_legend(fig, nomes)
    plt.subplots_adjust(bottom=0.07)
    plt.show()


def exportar_rede(store: Store, pred: dict, modelo: str, chaves: list[str]) -> None:
    """Salva as estradas previstas na prova final como linhas (abre no QGIS)."""
    import geopandas as gpd
    from skimage.morphology import skeletonize

    cfg, estrada, vet = store.cfg, store.cfg["classes"]["id"]["estrada"], []
    for k in chaves:
        for a, p in pred.get(("holdout", modelo, k), {}).items():
            g = aoi_grid(cfg, a, DATA_KEYS[k][1])
            linhas = MT.skeleton_to_lines(skeletonize(p == estrada), g.transform)
            vet.append(gpd.GeoDataFrame({"aoi": a, "dados": k, "modelo": modelo}, index=range(len(linhas)),
                                        geometry=linhas, crs=g.crs).to_crs(4326))
    if vet:
        pd.concat(vet, ignore_index=True).to_file(RESULTS / "rede_estradas_predita_holdout.gpkg", driver="GPKG")
        print("estradas previstas salvas em resultados/rede_estradas_predita_holdout.gpkg")


def avaliar_ouro(store: Store, pred: dict):
    """Métricas nos pontos conferidos à mão (só quando a coluna rotulo_manual estiver preenchida)."""
    cfg, nomes = store.cfg, store.cfg["classes"]["nomes"]
    ouro = L.load_gold(cfg)
    if ouro is None:
        print("Os pontos ainda não foram conferidos à mão: os resultados acima usam MapBiomas + OSM como rótulo.")
        return None
    linhas = []
    for (tipo, m, k), mapas in pred.items():
        if tipo == "blocos":
            continue
        r = DATA_KEYS[k][1]
        sub = ouro[ouro.aoi.isin(mapas)].copy()
        if sub.empty:
            continue
        prev = [mapas[a][int(l) * 10 // r, int(c) * 10 // r] for a, l, c in zip(sub.aoi, sub.linha, sub.coluna)]
        sub["pred"] = [nomes[v] if v != IGNORE else None for v in prev]
        s = MT.per_point_scores(sub, "pred", "rotulo_manual", nomes)
        linhas.append({"experimento": tipo, "modelo": m, "dados": k, "n_pontos": len(sub),
                       "f1_macro": s["f1_macro"], "f1_estrada": s["f1_estrada"]})
    df = pd.DataFrame(linhas)
    df.to_csv(RESULTS / "metricas_conjunto_ouro.csv", index=False)
    return df


def conclusoes(met: pd.DataFrame, comp) -> None:
    h = met[met.experimento == "holdout"]
    c = h.groupby("dados").completude.mean()
    print(f"1. Resolução: na prova final, os modelos encontram {100 * c.s2_10:.0f} % das estradas a 10 m, "
          f"{100 * c.s2_30:.0f} % a 30 m e {100 * c.s2_64:.0f} % a 64 m.")
    if "cbers_64" in c:
        print(f"2. Sensor real: com o CBERS de 64 m encontram {100 * c.cbers_64:.0f} %, contra {100 * c.s2_64:.0f} % "
              f"do Sentinel-2 reduzido para 64 m.")
    if comp is not None:
        print(f"3. Europa → Brasil: aprendendo só na Europa, o F1 médio cai de {comp.f1_macro_BR.mean():.2f} "
              f"para {comp.f1_macro_EU.mean():.2f}.")


def fontes_com_falha() -> pd.DataFrame:
    st = D.status_table()
    return st[~st.ok] if len(st) else st
