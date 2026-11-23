import streamlit as st
import plotly.graph_objects as go
from database import load_fleet_horizon
from optimizer import DEFAULT_PARAMS, solve_joint_milp
from llm_copilot import parse_disruption_prompt, generate_executive_debrief

st.set_page_config(page_title="AI Fleet Maintenance & Spare-Parts Copilot", layout="wide")

st.title("AI-Driven Fleet Maintenance & Spare-Parts Optimization Copilot")
st.markdown(
    "**Joint Predictive Maintenance + Supply Chain MILP + LLM Operations Copilot:** "
    "Co-optimizes preventative maintenance schedules and spare-parts freight under technician "
    "and multi-day lead-time constraints."
)

df_horizon, data_source = load_fleet_horizon()
st.caption(f"Active Telemetry Source: **{data_source}** | Fleet Size: **20 Turbofan Engines** | Horizon: **14 Days**")

for k, v in DEFAULT_PARAMS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "copilot_debrief" not in st.session_state:
    st.session_state["copilot_debrief"] = ""

# --- Sidebar: LLM Copilot ---
st.sidebar.header("LLM Plant Operations Copilot")

try:
    default_key = st.secrets.get("GROQ_API_KEY", "")
except Exception:
    default_key = ""

with st.sidebar.expander("LLM Engine Settings", expanded=not bool(default_key)):
    groq_key = st.text_input("Groq API Key", value=default_key, type="password", placeholder="gsk_...")

example_prompts = [
    "Custom scenario...",
    "Warehouse audit found 0 initial kits in stock, and customs delayed standard shipping to 5 days.",
    "One technician is out sick (max 1/day) and emergency breakdown penalties doubled to $36,000.",
    "Expedited air freight price spiked to $4,500 per kit due to fuel surcharges.",
]
preset = st.sidebar.selectbox("Quick Disruption Presets", example_prompts)
user_prompt = st.sidebar.text_area(
    "Describe an operational disruption:",
    value="" if preset == "Custom scenario..." else preset,
    height=85,
    placeholder="e.g., Standard shipping delayed to 5 days and only 1 technician available...",
)

col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("Apply Scenario", type="primary"):
    if user_prompt.strip():
        try:
            current_p = {k: st.session_state[k] for k in DEFAULT_PARAMS}
            before_res = solve_joint_milp(df_horizon, current_p)
            new_p, param_note = parse_disruption_prompt(user_prompt, current_p, groq_key)
            for k, v in new_p.items():
                st.session_state[k] = v
            after_res = solve_joint_milp(df_horizon, new_p)
            debrief = generate_executive_debrief(user_prompt, before_res, after_res, groq_key)
            st.session_state["copilot_debrief"] = f"**Parameter Adjustment:** {param_note}\n\n**Executive Debrief:** {debrief}"
        except Exception as e:
            st.sidebar.error(f"Copilot error: {e}")

if col_btn2.button("Reset Defaults"):
    for k, v in DEFAULT_PARAMS.items():
        st.session_state[k] = v
    st.session_state["copilot_debrief"] = ""

if st.session_state["copilot_debrief"]:
    st.info(st.session_state["copilot_debrief"])

# --- Sidebar: Manual Parameter Sliders ---
st.sidebar.markdown("---")
st.sidebar.subheader("Solver Constraints & Cost Parameters")
st.session_state["initial_stock"] = st.sidebar.slider("Initial Spare Kits in Warehouse", 0, 10, int(st.session_state["initial_stock"]))
st.session_state["max_techs_per_day"] = st.sidebar.slider("Max Technicians / Day", 1, 5, int(st.session_state["max_techs_per_day"]))
st.session_state["std_lead_time"] = st.sidebar.slider("Standard Freight Lead Time (Days)", 1, 7, int(st.session_state["std_lead_time"]))
st.session_state["cost_breakdown"] = st.sidebar.number_input("Unplanned Breakdown Penalty ($)", value=float(st.session_state["cost_breakdown"]), step=2000.0)
st.session_state["cost_kit_exp"] = st.sidebar.number_input("Expedited Air Kit Cost ($)", value=float(st.session_state["cost_kit_exp"]), step=200.0)

# --- Solve & Render Dashboard ---
result = solve_joint_milp(df_horizon, {k: st.session_state[k] for k in DEFAULT_PARAMS})

if result is None:
    st.error("Infeasible constraints! Increase technician capacity or initial spare kits in the sidebar.")
else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Joint ML-MILP Total Cost", f"${result['opt_cost']:,.0f}")
    c2.metric("Reactive Baseline Cost", f"${result['reactive_cost']:,.0f}")
    c3.metric("14-Day Net Savings", f"${result['savings']:,.0f}", f"{(result['savings']/result['reactive_cost'])*100:.1f}%")
    c4.metric("Expedited Kits Ordered", result["expedited_kits"])

    col_left, col_right = st.columns([1.25, 1])

    with col_left:
        st.subheader("14-Day Spare-Parts Ordering & Warehouse Inventory")
        pdf = result["parts_df"]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Standard Freight Orders", x=pdf["Day"], y=pdf["Standard Orders (Kits)"], marker_color="#2b5c8f"))
        fig.add_trace(go.Bar(name="Expedited Air Orders", x=pdf["Day"], y=pdf["Expedited Orders (Kits)"], marker_color="#e15759"))
        fig.add_trace(go.Scatter(name="Warehouse Stock (End of Day)", x=pdf["Day"], y=pdf["End-of-Day Warehouse Stock"], mode="lines+markers", line=dict(color="#59a14f", width=3)))
        fig.add_trace(go.Scatter(name="Engines Serviced", x=pdf["Day"], y=pdf["Engines Serviced"], mode="markers", marker=dict(color="#111111", size=10, symbol="diamond")))
        fig.update_layout(barmode="stack", xaxis_title="Planning Day (1–14)", yaxis_title="Kits / Engines", height=390)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Fleet Risk & Maintenance Schedule")
        sdf = result["sched_df"].sort_values("14-Day Failure Risk", ascending=False)
        st.dataframe(sdf[["Engine ID", "14-Day Failure Risk", "Scheduled PM Day"]], use_container_width=True, height=390)

    # --- Practical ERP Action Export ---
    st.subheader("ERP / SAP Dispatch & Purchase Order Plan")
    st.dataframe(result["parts_df"], use_container_width=True)
    csv_export = result["parts_df"].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download 14-Day Dispatch & Procurement Schedule (CSV)",
        data=csv_export,
        file_name="14day_maintenance_and_po_plan.csv",
        mime="text/csv",
    )
