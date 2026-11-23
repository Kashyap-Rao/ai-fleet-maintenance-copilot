import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

DEFAULT_PARAMS = {
    "initial_stock": 2,
    "max_techs_per_day": 2,
    "std_lead_time": 3,
    "exp_lead_time": 1,
    "cost_pm": 2500.0,
    "cost_breakdown": 18000.0,
    "cost_kit_std": 1200.0,
    "cost_kit_exp": 2400.0,
    "cost_holding": 80.0,
}

PARAM_BOUNDS = {
    "initial_stock": (0, 20),
    "max_techs_per_day": (1, 10),
    "std_lead_time": (1, 10),
    "exp_lead_time": (1, 5),
    "cost_pm": (500.0, 20000.0),
    "cost_breakdown": (2000.0, 150000.0),
    "cost_kit_std": (200.0, 15000.0),
    "cost_kit_exp": (500.0, 25000.0),
    "cost_holding": (10.0, 1000.0),
}

def sanitize_params(raw_params: dict) -> dict:
    """Clamps all solver parameters to physically valid operational bounds."""
    clean = {}
    for k, default_val in DEFAULT_PARAMS.items():
        val = raw_params.get(k, default_val)
        low, high = PARAM_BOUNDS[k]
        cast_type = type(default_val)
        clean[k] = cast_type(max(low, min(high, cast_type(val))))
    return clean

def solve_joint_milp(df_h: pd.DataFrame, params: dict) -> dict | None:
    """
    Jointly optimizes 14-day preventative maintenance scheduling (binary x[m, t])
    and spare-parts procurement with lead times (q_std[t], q_exp[t]).
    """
    p = sanitize_params(params)
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        raise RuntimeError("OR-Tools SCIP solver unavailable.")

    engines = sorted(df_h["engine_name"].unique())
    days = sorted(df_h["day"].unique())
    M, T = len(engines), len(days)

    prob_matrix = (
        df_h.pivot(index="engine_name", columns="day", values="cum_fail_prob")
        .loc[engines, days]
        .values
    )

    x = [[solver.IntVar(0, 1, f"x_{m}_{t}") for t in range(T)] for m in range(M)]
    q_std = [solver.IntVar(0, 20, f"q_std_{t}") for t in range(T)]
    q_exp = [solver.IntVar(0, 20, f"q_exp_{t}") for t in range(T)]
    inv = [solver.NumVar(0, 50, f"inv_{t}") for t in range(T)]

    for m in range(M):
        solver.Add(sum(x[m][t] for t in range(T)) <= 1)

    for t in range(T):
        solver.Add(sum(x[m][t] for m in range(M)) <= p["max_techs_per_day"])
        prev_inv = p["initial_stock"] if t == 0 else inv[t - 1]
        arr_std = q_std[t - p["std_lead_time"]] if t >= p["std_lead_time"] else 0
        arr_exp = q_exp[t - p["exp_lead_time"]] if t >= p["exp_lead_time"] else 0
        solver.Add(inv[t] == prev_inv + arr_std + arr_exp - sum(x[m][t] for m in range(M)))

    obj = solver.Objective()
    for t in range(T):
        obj.SetCoefficient(q_std[t], float(p["cost_kit_std"]))
        obj.SetCoefficient(q_exp[t], float(p["cost_kit_exp"]))
        obj.SetCoefficient(inv[t], float(p["cost_holding"]))

    for m in range(M):
        horizon_risk = float(prob_matrix[m, -1]) * p["cost_breakdown"]
        for t in range(T):
            risk_before = float(prob_matrix[m, t - 1]) * p["cost_breakdown"] if t > 0 else 0.0
            obj.SetCoefficient(x[m][t], float(p["cost_pm"] + risk_before - horizon_risk))

    offset = sum(float(prob_matrix[m, -1]) * p["cost_breakdown"] for m in range(M))
    obj.SetMinimization()

    if solver.Solve() != pywraplp.Solver.OPTIMAL:
        return None

    opt_cost = obj.Value() + offset

    sched_rows = []
    for m, eng in enumerate(engines):
        s_days = [days[t] for t in range(T) if x[m][t].solution_value() > 0.5]
        sched_rows.append({
            "Engine ID": eng,
            "14-Day Failure Risk": round(float(prob_matrix[m, -1]), 3),
            "Scheduled PM Day": f"Day {s_days[0]}" if s_days else "No PM Needed",
            "day_num": s_days[0] if s_days else None,
        })

    parts_rows = []
    for t, d in enumerate(days):
        serviced_list = [engines[m] for m in range(M) if x[m][t].solution_value() > 0.5]
        parts_rows.append({
            "Day": d,
            "Engines Serviced": len(serviced_list),
            "Assigned Engine IDs": ", ".join(serviced_list) if serviced_list else "-",
            "Standard Orders (Kits)": int(round(q_std[t].solution_value())),
            "Expedited Orders (Kits)": int(round(q_exp[t].solution_value())),
            "End-of-Day Warehouse Stock": round(inv[t].solution_value(), 1),
        })

    reactive_cost = 0.0
    for m in range(M):
        hi = np.where(prob_matrix[m] >= 0.65)[0]
        if len(hi) > 0:
            trig = min(T - 1, hi[0] + p["exp_lead_time"])
            reactive_cost += prob_matrix[m, trig] * p["cost_breakdown"] + p["cost_pm"] + p["cost_kit_exp"]
        else:
            reactive_cost += prob_matrix[m, -1] * p["cost_breakdown"]

    parts_df = pd.DataFrame(parts_rows)
    sched_df = pd.DataFrame(sched_rows)

    return {
        "opt_cost": opt_cost,
        "reactive_cost": reactive_cost,
        "savings": reactive_cost - opt_cost,
        "expedited_kits": int(parts_df["Expedited Orders (Kits)"].sum()),
        "standard_kits": int(parts_df["Standard Orders (Kits)"].sum()),
        "pm_count": int(parts_df["Engines Serviced"].sum()),
        "sched_df": sched_df,
        "parts_df": parts_df,
    }
