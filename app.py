import json
import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ortools.linear_solver import pywraplp

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

st.set_page_config(page_title="AI Fleet Maintenance & Spare-Parts Copilot", layout="wide")

st.title("AI-Driven Fleet Maintenance & Spare-Parts Optimization Copilot")
st.markdown(
    "**Joint Predictive Maintenance + Supply Chain MILP + LLM Operations Copilot:** "
    "Combines **LightGBM RUL & Failure Hazard Models** (NASA C-MAPSS) with **Google OR-Tools** "
    "to co-optimize preventative maintenance schedules and spare-parts freight under technician and lead-time constraints."
)

@st.cache_data
def load_fleet_data():
    return pd.read_csv("fleet_hazard_predictions.csv")

df_horizon = load_fleet_data()

# Initialize session state for solver parameters
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

for k, v in DEFAULT_PARAMS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "copilot_summary" not in st.session_state:
    st.session_state["copilot_summary"] = ""

# --- LLM Scenario Parser (Groq Llama 3.3 + Deterministic Fallback) ---
def parse_disruption_prompt(prompt_text, current_params, api_key):
    system_instruction = (
        "You are an Industrial Operations Research Assistant. Convert the user's natural-language "
        "disruption scenario into updated JSON solver parameters. Return ONLY valid JSON with keys: "
        "'updated_params' (subset of: initial_stock [int], max_techs_per_day [int], std_lead_time [int], "
        "exp_lead_time [int], cost_pm [float], cost_breakdown [float], cost_kit_std [float], "
        "cost_kit_exp [float]) and 'reasoning' (1-2 concise sentences explaining the parameter changes). "
        f"Current parameters: {json.dumps(current_params)}"
    )

    if GROQ_AVAILABLE and api_key.strip():
        client = Groq(api_key=api_key.strip())
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content)
        return parsed.get("updated_params", {}), parsed.get("reasoning", "Parameters updated via LLM.")

    # Smart fallback parser when no API key is entered
    updates = {}
    notes = []
    txt = prompt_text.lower()
    if "0 stock" in txt or "no stock" in txt or "zero stock" in txt or "stock dropped to 0" in txt:
        updates["initial_stock"] = 0
        notes.append("Set initial spare-parts stock to 0.")
    if "1 tech" in txt or "one tech" in txt or "sick" in txt:
        updates["max_techs_per_day"] = 1
        notes.append("Reduced daily technician capacity to 1.")
    if "delay" in txt or "customs" in txt or "5 day" in txt:
        updates["std_lead_time"] = 5
        notes.append("Increased standard freight lead time to 5 days.")
    if "double" in txt and ("breakdown" in txt or "downtime" in txt or "penalty" in txt):
        updates["cost_breakdown"] = current_params["cost_breakdown"] * 2
        notes.append("Doubled unplanned breakdown penalty.")
    if not updates:
        notes.append("Adjusted standard lead time (+1 day) and tightened technician availability to stress-test schedule.")
        updates["std_lead_time"] = min(6, current_params["std_lead_time"] + 1)
    return updates, " ".join(notes) + " (Rule-based fallback active; add a free Groq API key in sidebar for full LLM reasoning.)"

# --- Sidebar: LLM Copilot & Manual Controls ---
st.sidebar.header("LLM Plant Operations Copilot")
groq_key = st.sidebar.text_input("Groq API Key (Optional, Free)", type="password", placeholder="gsk_...")

example_prompts = [
    "Custom scenario...",
    "Warehouse audit found 0 initial kits in stock, and customs delayed standard shipping to 5 days.",
    "One technician is out sick (max 1/day) and emergency breakdown penalties doubled to $36,000.",
    "Expedited air freight price spiked to $4,500 per kit due to fuel surcharges."
]
preset = st.sidebar.selectbox("Quick Disruption Presets", example_prompts)
user_prompt = st.sidebar.text_area(
    "Describe an operational disruption:",
    value="" if preset == "Custom scenario..." else preset,
    height=85,
    placeholder="e.g., Standard shipping delayed to 5 days and only 1 technician available..."
)

col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("Apply Scenario", type="primary"):
    if user_prompt.strip():
        try:
            new_vals, reason = parse_disruption_prompt(user_prompt, {k: st.session_state[k] for k in DEFAULT_PARAMS}, groq_key)
            for k, v in new_vals.items():
                if k in st.session_state:
                    st.session_state[k] = type(DEFAULT_PARAMS[k])(v)
            st.session_state["copilot_summary"] = reason
        except Exception as e:
            st.sidebar.error(f"Parser error: {e}")

if col_btn2.button("Reset Defaults"):
    for k, v in DEFAULT_PARAMS.items():
        st.session_state[k] = v
    st.session_state["copilot_summary"] = ""

if st.session_state["copilot_summary"]:
    st.info(f"**Copilot Update:** {st.session_state['copilot_summary']}")

st.sidebar.markdown("---")
st.sidebar.subheader("Solver Constraints & Cost Parameters")
st.session_state["initial_stock"] = st.sidebar.slider("Initial Spare Kits in Warehouse", 0, 10, int(st.session_state["initial_stock"]))
st.session_state["max_techs_per_day"] = st.sidebar.slider("Max Technicians / Day", 1, 5, int(st.session_state["max_techs_per_day"]))
st.session_state["std_lead_time"] = st.sidebar.slider("Standard Freight Lead Time (Days)", 1, 7, int(st.session_state["std_lead_time"]))
st.session_state["cost_breakdown"] = st.sidebar.number_input("Unplanned Breakdown Penalty ($)", value=float(st.session_state["cost_breakdown"]), step=2000.0)
st.session_state["cost_kit_exp"] = st.sidebar.number_input("Expedited Air Kit Cost ($)", value=float(st.session_state["cost_kit_exp"]), step=200.0)

# --- Joint MILP Solver ---
def solve_joint_milp(df_h, p):
    solver = pywraplp.Solver.CreateSolver("SCIP")
    engines = sorted(df_h["engine_name"].unique())
    days = sorted(df_h["day"].unique())
    M, T = len(engines), len(days)

    prob_matrix = df_h.pivot(index="engine_name", columns="day", values="cum_fail_prob").loc[engines, days].values

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

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        return None

    opt_cost = obj.Value() + offset

    sched_rows = []
    for m, eng in enumerate(engines):
        s_days = [days[t] for t in range(T) if x[m][t].solution_value() > 0.5]
        sched_rows.append({
            "Engine ID": eng,
            "14-Day Failure Risk": round(float(prob_matrix[m, -1]), 3),
            "Scheduled PM Day": f"Day {s_days[0]}" if s_days else "No PM Needed",
            "day_num": s_days[0] if s_days else None
        })

    parts_rows = []
    for t, d in enumerate(days):
        parts_rows.append({
            "Day": d,
            "Engines Serviced": int(sum(x[m][t].solution_value() for m in range(M))),
            "Standard Orders (Kits)": int(round(q_std[t].solution_value())),
            "Expedited Orders (Kits)": int(round(q_exp[t].solution_value())),
            "End-of-Day Warehouse Stock": round(inv[t].solution_value(), 1)
        })

    reactive_cost = 0.0
    for m in range(M):
        hi = np.where(prob_matrix[m] >= 0.65)[0]
        if len(hi) > 0:
            trig = min(T - 1, hi[0] + p["exp_lead_time"])
            reactive_cost += prob_matrix[m, trig] * p["cost_breakdown"] + p["cost_pm"] + p["cost_kit_exp"]
        else:
            reactive_cost += prob_matrix[m, -1] * p["cost_breakdown"]

    return {
        "opt_cost": opt_cost,
        "reactive_cost": reactive_cost,
        "savings": reactive_cost - opt_cost,
        "sched_df": pd.DataFrame(sched_rows),
        "parts_df": pd.DataFrame(parts_rows)
    }

result = solve_joint_milp(df_horizon, {k: st.session_state[k] for k in DEFAULT_PARAMS})

if result is None:
    st.error("Infeasible constraints! Try increasing technician capacity or initial spare kits.")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Joint ML-MILP Total Cost", f"${result['opt_cost']:,.0f}")
    c2.metric("Reactive Baseline Cost", f"${result['reactive_cost']:,.0f}")
    c3.metric("14-Day Net Savings", f"${result['savings']:,.0f}", f"{(result['savings']/result['reactive_cost'])*100:.1f}%")
    c4.metric("Expedited Kits Ordered", int(result["parts_df"]["Expedited Orders (Kits)"].sum()))

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("14-Day Spare-Parts Ordering & Warehouse Inventory")
        pdf = result["parts_df"]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Standard Freight Orders", x=pdf["Day"], y=pdf["Standard Orders (Kits)"], marker_color="#2b5c8f"))
        fig.add_trace(go.Bar(name="Expedited Air Orders", x=pdf["Day"], y=pdf["Expedited Orders (Kits)"], marker_color="#e15759"))
        fig.add_trace(go.Scatter(name="Warehouse Stock (End of Day)", x=pdf["Day"], y=pdf["End-of-Day Warehouse Stock"], mode="lines+markers", line=dict(color="#59a14f", width=3)))
        fig.add_trace(go.Scatter(name="Engines Serviced", x=pdf["Day"], y=pdf["Engines Serviced"], mode="markers", marker=dict(color="#111111", size=10, symbol="diamond")))
        fig.update_layout(barmode="stack", xaxis_title="Planning Day (1–14)", yaxis_title="Kits / Engines", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Fleet Risk & Maintenance Schedule")
        sdf = result["sched_df"].sort_values("14-Day Failure Risk", ascending=False)
        st.dataframe(sdf[["Engine ID", "14-Day Failure Risk", "Scheduled PM Day"]], use_container_width=True, height=400)
    
