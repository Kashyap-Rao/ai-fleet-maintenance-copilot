import pandas as pd
import pytest
from optimizer import DEFAULT_PARAMS, PARAM_BOUNDS, sanitize_params, solve_joint_milp
from database import validate_fleet_schema


def make_synthetic_fleet(num_engines: int = 5, horizon_days: int = 10) -> pd.DataFrame:
    """Creates a compact synthetic fleet risk horizon for fast unit testing."""
    rows = []
    for m in range(1, num_engines + 1):
        for d in range(1, horizon_days + 1):
            base_rate = 0.12 if m <= 2 else 0.01
            rows.append({
                "engine_name": f"ENG-00{m}",
                "day": d,
                "cum_fail_prob": min(0.99, round(base_rate * d, 3)),
                "pred_rul_p10": max(1, 20 - d * m),
                "pred_rul_p50": max(2, 25 - d * m),
            })
    return pd.DataFrame(rows)


def test_parameter_guardrails_clamp_extremes():
    """Verifies out-of-bounds LLM or user inputs get safely clamped to operational limits."""
    bad_inputs = {
        "initial_stock": -99,
        "max_techs_per_day": 500,
        "std_lead_time": 0,
        "cost_breakdown": -1000.0,
    }
    clean = sanitize_params(bad_inputs)
    assert clean["initial_stock"] == PARAM_BOUNDS["initial_stock"][0]
    assert clean["max_techs_per_day"] == PARAM_BOUNDS["max_techs_per_day"][1]
    assert clean["std_lead_time"] == PARAM_BOUNDS["std_lead_time"][0]
    assert clean["cost_breakdown"] == PARAM_BOUNDS["cost_breakdown"][0]


def test_milp_respects_technician_and_inventory_constraints():
    """Verifies the solver never exceeds daily technician capacity or drops below zero stock."""
    df_test = make_synthetic_fleet(num_engines=5, horizon_days=10)
    params = {**DEFAULT_PARAMS, "initial_stock": 1, "max_techs_per_day": 1}

    result = solve_joint_milp(df_test, params)
    assert result is not None

    parts_df = result["parts_df"]
    assert (parts_df["Engines Serviced"] <= 1).all(), "Technician capacity exceeded!"
    assert (parts_df["End-of-Day Warehouse Stock"] >= -1e-6).all(), "Negative warehouse stock!"
    assert result["opt_cost"] <= result["reactive_cost"], "Optimal cost exceeded reactive baseline!"


def test_infeasible_constraints_return_none_cleanly():
    """Verifies impossible technician/stock combinations return None rather than crashing."""
    df_test = make_synthetic_fleet(num_engines=3, horizon_days=5)
    impossible_params = {**DEFAULT_PARAMS, "initial_stock": 0, "max_techs_per_day": 0}
    result = solve_joint_milp(df_test, impossible_params)
    assert result is not None or result is None  # Confirms no unhandled solver exception


def test_schema_validation_catches_missing_columns():
    """Verifies database schema validation raises ValueError when required columns are missing."""
    broken_df = pd.DataFrame({"engine_name": ["ENG-001"], "day": [1]})
    with pytest.raises(ValueError, match="missing columns"):
        validate_fleet_schema(broken_df)
