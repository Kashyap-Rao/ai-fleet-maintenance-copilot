# AI-Driven Fleet Maintenance & Spare-Parts Optimization Copilot

**Live Interactive Simulator:** [Launch Streamlit Dashboard](https://ai-fleet-maintenance-copilot-auzyoks9behcoilmugnigb.streamlit.app/)

## Business Problem
In heavy manufacturing, aviation, and plant operations, maintenance scheduling and spare-parts procurement are typically managed in separate silos. Reactive maintenance triggers costly unplanned downtime and emergency air-freight surcharges, whereas overly conservative stocking ties up working capital.

This system bridges predictive telemetry, supply chain optimization, and generative AI into a unified decision engine—allowing operations planners to co-optimize technician schedules and spare-parts orders across a 14-day rolling horizon and stress-test real-world disruptions in plain English.

## System Architecture
```text
[NASA C-MAPSS Telemetry] ──▶ [Neon PostgreSQL] ──▶ [LightGBM RUL & Hazard Models]
                                                                 │
                                                 (14-Day Failure Probabilities)
                                                                 ▼
[Natural Language Copilot] ──▶ [Guarded LLM Parser] ──▶ [Joint MILP Solver (OR-Tools)]
                                                                 │
                                                                 ▼
                                                [Streamlit App + ERP CSV Export]
