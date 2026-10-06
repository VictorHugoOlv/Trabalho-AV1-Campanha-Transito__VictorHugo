from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from .data_processing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_data, feature_frame, target

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
REPORTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)


def build_pipeline(max_depth=6, min_samples_leaf=20):
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    preprocessor = ColumnTransformer([
        ("cat", categorical, CATEGORICAL_FEATURES),
        ("num", numeric, NUMERIC_FEATURES)
    ])

    classifier = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=42
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])


def train_and_evaluate():
    df = load_data()
    X = feature_frame(df)
    y = target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    metrics = {
        "amostras_total": int(len(df)),
        "treino": int(len(X_train)),
        "teste": int(len(X_test)),
        "acuracia": float(accuracy_score(y_test, pred)),
        "precisao": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "classes": {"0": "Sem vítimas", "1": "Com vítimas"}
    }

    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    report = classification_report(y_test, pred, target_names=["Sem vítimas", "Com vítimas"], zero_division=0)
    (REPORTS / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Sem vítimas", "Com vítimas"], yticklabels=["Sem vítimas", "Com vítimas"])
    plt.title("Matriz de confusão — Árvore de Decisão")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(REPORTS / "confusion_matrix.png", dpi=160)
    plt.close()

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["classifier"].feature_importances_
    fi = pd.DataFrame({"atributo": feature_names, "importancia": importances})
    fi = fi.sort_values("importancia", ascending=False)
    fi.to_csv(REPORTS / "feature_importance.csv", index=False, encoding="utf-8-sig")

    top = fi.head(15).sort_values("importancia")
    plt.figure(figsize=(9, 6))
    plt.barh(top["atributo"], top["importancia"])
    plt.title("15 atributos com maior importância")
    plt.xlabel("Importância")
    plt.tight_layout()
    plt.savefig(REPORTS / "feature_importance.png", dpi=160)
    plt.close()

    joblib.dump(pipeline, MODELS / "decision_tree.joblib")
    return pipeline, metrics, fi


if __name__ == "__main__":
    pipeline, metrics, fi = train_and_evaluate()
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print("\nPrincipais atributos:")
    print(fi.head(15).to_string(index=False))
