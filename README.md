# Trabalho AV1 — Campanha de Trânsito

Projeto completo da AV1 de Sistemas de Apoio à Tomada de Decisão.

## Objetivo

Aplicar uma **Árvore de Decisão baseada em entropia e ganho de informação** para identificar atributos associados ao resultado dos sinistros de trânsito de 2025 e apoiar o direcionamento de campanhas educativas.

A variável-alvo foi construída a partir dos dados de pessoas:

- **Com vítimas**: `mortos + feridos > 0`
- **Sem vítimas**: `mortos + feridos = 0`

As variáveis de resultado (`mortos`, `feridos`, `feridos_leves`, `feridos_graves`, `ilesos`, `ignorados`, `classificacao_acidente`) não são usadas como atributos de entrada, evitando vazamento da resposta para o modelo.

## Estrutura

```text
campanha_transito/
├── app.py                         # Interface Streamlit
├── requirements.txt
├── README.md
├── data/
│   └── datatran2025.csv
├── src/
│   ├── __init__.py
│   ├── data_processing.py         # leitura e tratamento
│   ├── model.py                   # treinamento e avaliação
│   └── analysis.py                # gráficos e análises
├── models/                        # modelo salvo após execução
├── reports/                       # métricas, gráficos e importâncias
└── tests/
    └── test_pipeline.py
```

## Instalação

Python 3.10+ recomendado.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Depois:

```bash
pip install -r requirements.txt
```

## Executar a análise

```bash
python -m src.model
```

Isso gera:

- `reports/metrics.json`
- `reports/classification_report.txt`
- `reports/confusion_matrix.png`
- `reports/feature_importance.png`
- `models/decision_tree.joblib`

## Executar a interface

```bash
streamlit run app.py
```

A interface permite informar características de um novo sinistro, obter a classificação prevista, visualizar probabilidades e consultar os principais atributos utilizados pelo modelo.

## Métricas

São apresentadas:

- Acurácia
- Precisão
- Recall
- F1-score
- Matriz de confusão

O conjunto de teste é separado de forma estratificada, mantendo a proporção das classes.

## Observação acadêmica

O modelo é uma ferramenta de apoio à decisão e não deve ser interpretado como causalidade. Importância de variável indica contribuição para a previsão do modelo, não prova de que determinada característica cause a ocorrência de vítimas.
