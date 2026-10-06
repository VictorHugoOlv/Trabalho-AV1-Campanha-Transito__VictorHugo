from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

from src.data_processing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_data
from src.model import train_and_evaluate

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "decision_tree.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
FI_PATH = ROOT / "reports" / "feature_importance.csv"

st.set_page_config(page_title="Campanha de Trânsito — SAD", page_icon="🚦", layout="wide")
st.title("🚦 Sistema de Apoio à Decisão — Campanhas Educativas no Trânsito")
st.caption("Árvore de Decisão baseada em entropia e ganho de informação")

@st.cache_resource
def get_model():
    if not MODEL_PATH.exists():
        train_and_evaluate()
    return joblib.load(MODEL_PATH)

@st.cache_data
def get_data():
    return load_data()

model = get_model()
df = get_data()

with st.sidebar:
    st.header("Dados do projeto")
    st.metric("Sinistros analisados", f"{len(df):,}".replace(",", "."))
    st.metric("Com vítimas", f"{int(df.resultado_sinistro.sum()):,}".replace(",", "."))
    st.metric("Sem vítimas", f"{int((df.resultado_sinistro == 0).sum()):,}".replace(",", "."))

aba1, aba2, aba3 = st.tabs(["🔎 Previsão", "📊 Métricas", "🌳 Importância dos atributos"])

with aba1:
    st.subheader("Classificação de um novo sinistro")
    col1, col2 = st.columns(2)

    with col1:
        uf = st.selectbox("UF", sorted(df["uf"].dropna().unique()))
        dia = st.selectbox("Dia da semana", sorted(df["dia_semana"].dropna().unique()))
        causa = st.selectbox("Causa do acidente", sorted(df["causa_acidente"].dropna().unique()))
        tipo = st.selectbox("Tipo de acidente", sorted(df["tipo_acidente"].dropna().unique()))
        fase = st.selectbox("Fase do dia", sorted(df["fase_dia"].dropna().unique()))

    with col2:
        sentido = st.selectbox("Sentido da via", sorted(df["sentido_via"].dropna().unique()))
        clima = st.selectbox("Condição meteorológica", sorted(df["condicao_metereologica"].dropna().unique()))
        pista = st.selectbox("Tipo de pista", sorted(df["tipo_pista"].dropna().unique()))
        tracado = st.selectbox("Traçado da via", sorted(df["tracado_via"].dropna().unique()))
        uso_solo = st.selectbox("Uso do solo", sorted(df["uso_solo"].dropna().unique()))

    col3, col4, col5, col6, col7 = st.columns(5)
    with col3:
        br = st.number_input("BR", min_value=0.0, value=float(df.br.median()))
    with col4:
        km = st.number_input("KM", min_value=0.0, value=float(df.km.median()))
    with col5:
        pessoas = st.number_input("Pessoas", min_value=1, value=int(df.pessoas.median()))
    with col6:
        veiculos = st.number_input("Veículos", min_value=1, value=int(df.veiculos.median()))
    with col7:
        hora = st.slider("Hora", 0, 23, 12)

    if st.button("Analisar sinistro", type="primary"):
        novo = pd.DataFrame([{
            "uf": uf, "dia_semana": dia, "causa_acidente": causa,
            "tipo_acidente": tipo, "fase_dia": fase, "sentido_via": sentido,
            "condicao_metereologica": clima, "tipo_pista": pista,
            "tracado_via": tracado, "uso_solo": uso_solo,
            "br": br, "km": km, "pessoas": pessoas, "veiculos": veiculos, "hora": hora
        }])
        pred = int(model.predict(novo)[0])
        prob = model.predict_proba(novo)[0]
        label = "Com vítimas" if pred == 1 else "Sem vítimas"

        if pred == 1:
            st.error(f"Resultado previsto: **{label}**")
            st.warning("Recomendação: priorizar ações educativas/preventivas e reforçar estratégias voltadas às características apresentadas pelo sinistro.")
        else:
            st.success(f"Resultado previsto: **{label}**")
            st.info("Recomendação: manter monitoramento e utilizar o perfil junto aos demais resultados agregados para definição das campanhas.")

        st.write(f"Probabilidade estimada de **sem vítimas**: {prob[0]:.1%}")
        st.write(f"Probabilidade estimada de **com vítimas**: {prob[1]:.1%}")

with aba2:
    st.subheader("Desempenho do modelo")
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        cols = st.columns(4)
        cols[0].metric("Acurácia", f"{metrics['acuracia']:.2%}")
        cols[1].metric("Precisão", f"{metrics['precisao']:.2%}")
        cols[2].metric("Recall", f"{metrics['recall']:.2%}")
        cols[3].metric("F1-score", f"{metrics['f1']:.2%}")

    img = REPORTS = ROOT / "reports" / "confusion_matrix.png"
    if img.exists():
        st.image(str(img), caption="Matriz de confusão")

with aba3:
    st.subheader("Atributos mais relevantes para o modelo")
    if FI_PATH.exists():
        fi = pd.read_csv(FI_PATH).head(15).sort_values("importancia")
        st.bar_chart(fi.set_index("atributo")["importancia"])
        st.dataframe(fi, use_container_width=True, hide_index=True)
        st.caption("Importância de atributo indica contribuição para a previsão do modelo; não representa causalidade.")
