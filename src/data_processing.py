from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "datatran2025.csv"

CATEGORICAL_FEATURES = [
    "uf", "dia_semana", "causa_acidente", "tipo_acidente",
    "fase_dia", "sentido_via", "condicao_metereologica",
    "tipo_pista", "tracado_via", "uso_solo"
]

NUMERIC_FEATURES = ["br", "km", "pessoas", "veiculos", "hora"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, sep=";", encoding="latin1", low_memory=False)
    return clean_data(df)


def clean_data(df):
    df = df.copy()

    # Padroniza textos e transforma valores vazios em ausentes.
    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"nan": pd.NA, "": pd.NA})

    # Números brasileiros e valores com vírgula decimal.
    for col in ["br", "km", "pessoas", "mortos", "feridos_leves", "feridos_graves",
                "ilesos", "ignorados", "feridos", "veiculos"]:
        if col in df:
            df[col] = pd.to_numeric(
                df[col].astype("string").str.replace(",", ".", regex=False),
                errors="coerce"
            )

    # Horário -> hora do dia.
    df["horario"] = pd.to_datetime(df["horario"], format="%H:%M:%S", errors="coerce")
    df["hora"] = df["horario"].dt.hour

    # Alvo solicitado no enunciado: com vítimas (feridas ou fatais) / sem vítimas.
    df["resultado_sinistro"] = ((df["mortos"].fillna(0) + df["feridos"].fillna(0)) > 0).astype(int)
    df["resultado_label"] = df["resultado_sinistro"].map({0: "Sem vítimas", 1: "Com vítimas"})

    # Removemos linhas sem alvo.
    df = df.dropna(subset=["resultado_sinistro"])
    return df


def feature_frame(df):
    return df[FEATURES].copy()


def target(df):
    return df["resultado_sinistro"].astype(int)
