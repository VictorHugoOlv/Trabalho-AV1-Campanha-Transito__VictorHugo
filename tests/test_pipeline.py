import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_processing import load_data, FEATURES
from src.model import build_pipeline


def test_target_is_created():
    df = load_data()
    assert "resultado_sinistro" in df.columns
    assert set(df["resultado_sinistro"].unique()).issubset({0, 1})


def test_features_exist():
    df = load_data()
    assert all(col in df.columns for col in FEATURES)


def test_pipeline_can_be_built():
    assert build_pipeline() is not None
