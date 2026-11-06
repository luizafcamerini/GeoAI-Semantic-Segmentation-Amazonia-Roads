# Classificação de terreno em séries temporais de baixa resolução (foco em estradas, validação no Brasil)

Notebook principal: `classificacao_terreno_br.ipynb`. Parâmetros: `config.yaml`. Funções: `src/`.

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv -r requirements.txt
.venv/Scripts/python -m ipykernel install --user --name classificacao-terreno-br   # opcional
# modo rápido (padrão), de ponta a ponta:
.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 classificacao_terreno_br.ipynb
# testes de vazamento (rodar depois do notebook para checar os splits reais):
.venv/Scripts/python -m pytest -q tests
```

- `MODO_RAPIDO=0` (variável de ambiente) ou `projeto.modo_rapido: false` no `config.yaml` ativa o modo completo.
- Downloads ficam em cache em `data/` (nada é baixado de novo). As saídas vão para `resultados/`.
- O conjunto ouro para interpretação manual fica em `validacao/pontos_para_interpretacao.csv|.gpkg`. Preencha `rotulo_manual`
  com um destes valores: `outros, agua, vegetacao_baixa, floresta, construcoes, estrada`, e execute o notebook de novo.
- Nenhuma credencial é necessária. O Earth Engine é opcional e fica desligado.
