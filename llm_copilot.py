import json
from optimizer import PARAM_BOUNDS, sanitize_params

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

def get_active_groq_model(client) -> str:
    models = [m.id for m in client.models.list().data]
    preferred = [
        "llama-3.1-8b-instant",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "qwen/qwen3-32b",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
    ]
    for p in preferred:
        if p in models:
            return p
    for m in models:
        if not any(skip in m.lower() for skip in ["whisper", "guard", "tts", "embed"]):
            return m
    raise RuntimeError(f"No active chat models available: {models}")

def parse_disruption_prompt(prompt_text: str, current_params: dict, api_key: str) -> tuple[dict, str]:
    """Translates plain-English operational disruptions into bounded, validated MILP parameters."""
    system_instruction = (
        "You are an Industrial Operations Research Assistant. Convert the user's disruption scenario "
        "into updated JSON solver parameters. Return ONLY a raw JSON object with keys: "
        "'updated_params' (subset of: initial_stock [int], max_techs_per_day [int], std_lead_time [int], "
        "exp_lead_time [int], cost_pm [float], cost_breakdown [float], cost_kit_std [float], "
        f"cost_kit_exp [float]) and 'reasoning' (1 concise sentence). Allowed bounds: {json.dumps(PARAM_BOUNDS)}. "
        f"Current parameters: {json.dumps(current_params)}"
    )

    if GROQ_AVAILABLE and api_key.strip():
        client = Groq(api_key=api_key.strip())
        model_id = get_active_groq_model(client)
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content)
        merged = {**current_params, **parsed.get("updated_params", {})}
        return sanitize_params(merged), parsed.get("reasoning", "Parameters updated.")

    # Deterministic fallback when no API key is configured
    merged = current_params.copy()
    notes = []
    txt = prompt_text.lower()
    if any(k in txt for k in ["0 stock", "no stock", "zero stock", "0 initial"]):
        merged["initial_stock"] = 0
        notes.append("Set initial stock to 0.")
    if any(k in txt for k in ["1 tech", "one tech", "sick"]):
        merged["max_techs_per_day"] = 1
        notes.append("Reduced technician capacity to 1/day.")
    if any(k in txt for k in ["delay", "customs", "5 day"]):
        merged["std_lead_time"] = 5
        notes.append("Increased standard lead time to 5 days.")
    if "double" in txt and any(k in txt for k in ["breakdown", "downtime", "penalty"]):
        merged["cost_breakdown"] = current_params["cost_breakdown"] * 2
        notes.append("Doubled breakdown penalty.")
    if not notes:
        merged["std_lead_time"] = min(6, current_params["std_lead_time"] + 1)
        notes.append("Increased standard lead time (+1 day).")
    return sanitize_params(merged), " ".join(notes)

def generate_executive_debrief(prompt_text: str, before_res: dict | None, after_res: dict | None, api_key: str) -> str:
    """Generates a post-solve operational explanation comparing before vs. after MILP KPIs."""
    if after_res is None:
        return "Solver Warning: Constraints are now mathematically infeasible. Increase technician capacity or initial spare kits."

    if not before_res:
        return "Schedule updated."

    cost_diff = after_res["opt_cost"] - before_res["opt_cost"]
    exp_diff = after_res["expedited_kits"] - before_res["expedited_kits"]
    fallback_summary = (
        f"Optimal 14-day fleet cost shifted by ${cost_diff:+,.0f} (to ${after_res['opt_cost']:,.0f}), "
        f"with {after_res['expedited_kits']} expedited air kits ({exp_diff:+d}) and "
        f"{after_res['standard_kits']} standard freight kits scheduled across {after_res['pm_count']} maintenance events."
    )

    if GROQ_AVAILABLE and api_key.strip():
        try:
            client = Groq(api_key=api_key.strip())
            model_id = get_active_groq_model(client)
            debrief_prompt = (
                f"Disruption scenario: '{prompt_text}'.\n"
                f"Before KPIs: Cost=${before_res['opt_cost']:,.0f}, StdKits={before_res['standard_kits']}, ExpKits={before_res['expedited_kits']}.\n"
                f"After KPIs: Cost=${after_res['opt_cost']:,.0f}, StdKits={after_res['standard_kits']}, ExpKits={after_res['expedited_kits']}.\n"
                "Write a crisp 2-sentence plant manager debrief explaining WHY the solver shifted freight modes or maintenance timing."
            )
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": debrief_prompt}],
                temperature=0.2,
                max_tokens=140,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return fallback_summary

    return fallback_summary
