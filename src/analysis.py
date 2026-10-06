from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .data_processing import load_data

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def generate_exploratory_graphs():
    df = load_data()
    REPORTS.mkdir(exist_ok=True)

    # Distribuição do resultado.
    counts = df["resultado_label"].value_counts()
    plt.figure(figsize=(7, 5))
    counts.reindex(["Sem vítimas", "Com vítimas"]).plot(kind="bar")
    plt.title("Distribuição dos sinistros por resultado")
    plt.ylabel("Quantidade")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(REPORTS / "resultado_distribuicao.png", dpi=160)
    plt.close()

    # Top causas por quantidade de sinistros com vítimas.
    top = (
        df[df["resultado_sinistro"] == 1]["causa_acidente"]
        .value_counts().head(10).sort_values()
    )
    plt.figure(figsize=(10, 6))
    top.plot(kind="barh")
    plt.title("10 principais causas entre sinistros com vítimas")
    plt.xlabel("Quantidade de sinistros")
    plt.tight_layout()
    plt.savefig(REPORTS / "top_causas_vitimas.png", dpi=160)
    plt.close()

    return counts
