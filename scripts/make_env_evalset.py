import json, random, os
random.seed(0)

SYSTEM = "You are an environmental engineering expert."

# 30 prompts: cover feedstock, inhibitors, temperature regimes, counterfactual, stability, monitoring.
PROMPTS = [
  "Engineer runs CSTR dark fermentation on food waste. Provide operating checklist table.",
  "Thermophilic DF reactor (55C) shows VFA spike; provide control checklist table.",
  "OLR increased 2x, H2 yield dropped; provide checklist table with actions and monitoring.",
  "pH drifting down to 4.8; provide checklist table with mitigation and risks.",
  "Reactor treats molasses wastewater; include COD, nutrients, trace metals checklist table.",
  "High ammonia suspected; include NH3 inhibition and control actions table.",
  "Sulfide/H2S detected; include sulfide inhibition and control actions table.",
  "Salinity increased (brackish feed); include salinity inhibition table.",
  "O2 intrusion from leaking seal; include ORP/O2 intrusion response table.",
  "Short HRT causes washout; provide table emphasizing biomass retention and HRT tuning.",
  "Long HRT triggers methanogenesis; provide table emphasizing suppression actions.",
  "Inoculum pretreatment options (heat shock/acid/base) and when to use; include in table.",
  "Mixing too strong causes shear; include mixing range and actions table.",
  "Trace metal deficiency suspected (Fe/Ni/Co); include supplementation in table.",
  "Low buffering; alkalinity insufficient; include VFA/alkalinity control in table.",
  "Startup phase: what ranges to target first week vs steady state; table must remain single.",
  "Counterfactual: if pH 5.5->6.2, how changes and actions; capture in high/low effects.",
  "Counterfactual: temperature 35->55C, implications; include actions.",
  "Feedstock shifts from glucose to lignocellulosic hydrolysate; include inhibitors and COD.",
  "Foaming observed; include practical controls (mixing, antifoam) table.",
  "ORP rises to -50 mV; include interpretation and actions table.",
  "OLR too low; productivity low; include actions table.",
  "Gas composition shows CH4; include methanogenesis suppression table.",
  "VFA/alk ratio high; include stable-operation responses table.",
  "N/P ratio off; include nutrient balancing table.",
  "Heavy metals present (Cu/Zn); include control actions table.",
  "pH control via NaHCO3 vs CaCO3; include in actions table.",
  "Online sensors available: pH, ORP, temp, gas flow; include monitoring methods in table.",
  "Batch DF reactor; include HRT equivalent (batch time) guidance in table.",
  "Scale-up: mixing/heat removal constraints; include in table."
]

TEMPLATE = """You are advising an engineer operating an anaerobic dark fermentation H2 reactor.

Return ONLY one Markdown pipe table with columns:
| Parameter | Typical target range (with units) | Monitoring | High/low effects | Practical control actions |

Hard rules:
- Every table row MUST start with '|' and end with '|'.
- Use ONLY plain text. No HTML tags.
- Each cell <= 18 words.
- Do NOT use “Varies by ...”.
- Must include at least: pH, temperature, HRT, OLR, substrate/COD, inoculum pretreatment, ORP,
  VFA/alkalinity, nutrients (N,P,trace metals), mixing, inhibitors (NH3, sulfide, heavy metals, salinity, O2 intrusion).
- No text before/after the table.
- End with a single line: END

Scenario: {scenario}
"""

os.makedirs("data/eval", exist_ok=True)
out_path = "data/eval/env_table_eval.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for i, sc in enumerate(PROMPTS):
        f.write(json.dumps({
            "id": f"env_table_{i:03d}",
            "system": SYSTEM,
            "prompt": TEMPLATE.format(scenario=sc)
        }, ensure_ascii=False) + "\n")
print("Wrote", out_path, "N=", len(PROMPTS))
