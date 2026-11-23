import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

REQUIRED_COLUMNS = {"engine_name", "day", "cum_fail_prob", "pred_rul_p10", "pred_rul_p50"}

def validate_fleet_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Enforces schema and data-quality guardrails before passing data to the MILP solver."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Data quality check failed — missing columns: {missing}")
    if df.empty:
        raise ValueError("Data quality check failed — dataset contains 0 rows.")
    df = df.copy()
    df["cum_fail_prob"] = df["cum_fail_prob"].clip(0.0, 1.0)
    df["day"] = df["day"].astype(int)
    return df

@st.cache_data(ttl=600)
def load_fleet_horizon() -> tuple[pd.DataFrame, str]:
    """
    Loads 14-day fleet hazard predictions from Neon PostgreSQL if DATABASE_URL is configured
    in Streamlit Secrets; otherwise falls back cleanly to the local CSV artifact.
    """
    db_url = ""
    try:
        db_url = st.secrets.get("DATABASE_URL", "")
    except Exception:
        db_url = ""

    if db_url.strip():
        try:
            # Normalize scheme for SQLAlchemy 2.x + psycopg3
            clean_url = db_url.strip().replace("postgresql://", "postgresql+psycopg://", 1)
            engine = create_engine(clean_url, pool_pre_ping=True)
            with engine.connect() as conn:
                df = pd.read_sql(text("SELECT * FROM fleet_hazard_predictions"), conn)
            return validate_fleet_schema(df), "Neon PostgreSQL (Live Cloud DB)"
        except Exception as e:
            st.warning(f"Neon DB unreachable ({e}); falling back to local CSV snapshot.")

    df = pd.read_csv("fleet_hazard_predictions.csv")
    return validate_fleet_schema(df), "Local CSV Artifact (Fallback)"
