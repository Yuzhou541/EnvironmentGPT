#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="envgpt"
POLITE_DELAY="1.0"
TIMEOUT="60"
MAX_RESULTS="2000"

run_one () {
  local tag="$1"
  local query="$2"
  local out_dir="$3"
  local meta_out="$4"

  mkdir -p "$out_dir" "$(dirname "$meta_out")" logs

  local ts log
  ts="$(date +%Y%m%d_%H%M%S)"
  log="logs/openalex_${tag}_${ts}.log"

  echo "============================================================"
  echo "[START] ${tag}"
  echo "  out_dir = ${out_dir}"
  echo "  meta    = ${meta_out}"
  echo "  log     = ${log}"
  echo "  query   = ${query}"
  echo "============================================================"

  conda run -n "${ENV_NAME}" python -u -m src.data.oa_pdf_crawler \
    --mode openalex \
    --query "${query}" \
    --max_results "${MAX_RESULTS}" \
    --out_dir "${out_dir}" \
    --meta_out "${meta_out}" \
    --polite_delay "${POLITE_DELAY}" \
    --timeout "${TIMEOUT}" \
    2>&1 | tee "${log}"

  echo "[DONE] ${tag}"
}

# q1_overview (你要求补上)
run_one "q1_overview" \
'"dark fermentation" ("food waste" OR "kitchen waste" OR OFMSW OR "organic fraction") (biohydrogen OR "hydrogen production" OR "H2 production")' \
"data/pdfs/openalex_q1_overview" \
"data/processed/openalex_q1_overview.jsonl"

run_one "q2_pretreatment" \
'"dark fermentation" ("food waste" OR "kitchen waste" OR OFMSW OR "organic fraction") (pretreatment OR "thermal pretreatment" OR "acid pretreatment" OR alkaline OR enzymatic OR ultrasound OR microwave OR "steam explosion") (biohydrogen OR "hydrogen production" OR "H2 production")' \
"data/pdfs/openalex_q2_pretreatment" \
"data/processed/openalex_q2_pretreatment.jsonl"

run_one "q3_inoculum" \
'"dark fermentation" ("food waste" OR "kitchen waste" OR OFMSW OR "organic fraction") (inoculum OR "anaerobic sludge" OR "heat shock" OR "acid treatment" OR "inoculum pretreatment" OR methanogen) (biohydrogen OR "hydrogen production")' \
"data/pdfs/openalex_q3_inoculum" \
"data/processed/openalex_q3_inoculum.jsonl"

run_one "q4_operating_parameters" \
'"dark fermentation" (biohydrogen OR "hydrogen production") ("food waste" OR "kitchen waste" OR OFMSW OR "organic fraction") (pH OR temperature OR HRT OR OLR OR "C/N" OR agitation OR ORP OR "hydrogen partial pressure")' \
"data/pdfs/openalex_q4_operating" \
"data/processed/openalex_q4_operating.jsonl"

run_one "q5_reactor_process" \
'"dark fermentation" ("food waste" OR OFMSW OR "organic fraction") (CSTR OR UASB OR "packed bed" OR "fluidized bed" OR "membrane bioreactor" OR "sequencing batch" OR continuous OR "two-stage") (biohydrogen OR "hydrogen production")' \
"data/pdfs/openalex_q5_reactor" \
"data/processed/openalex_q5_reactor.jsonl"

run_one "q6_codigestion" \
'("dark fermentation" OR "anaerobic fermentation") (biohydrogen OR "hydrogen production") ("food waste" OR OFMSW) ("co-digestion" OR "co-fermentation" OR "co-substrate")' \
"data/pdfs/openalex_q6_codigestion" \
"data/processed/openalex_q6_codigestion.jsonl"

run_one "q7_microbiome_pathway" \
'"dark fermentation" ("food waste" OR OFMSW OR "organic fraction") (Clostridium OR Enterobacter OR "microbial community" OR metagenomics OR "16S rRNA" OR pathway OR "butyrate-type" OR "acetate-type") (biohydrogen OR "hydrogen production")' \
"data/pdfs/openalex_q7_microbiome" \
"data/processed/openalex_q7_microbiome.jsonl"

run_one "q8_kinetics_modeling" \
'("dark fermentation" OR biohydrogen) ("food waste" OR OFMSW) (kinetic OR kinetics OR "modified Gompertz" OR Gompertz OR "first-order" OR "hydrogen production rate")' \
"data/pdfs/openalex_q8_kinetics" \
"data/processed/openalex_q8_kinetics.jsonl"

run_one "q9_inhibition_stability" \
'"dark fermentation" ("food waste" OR OFMSW) (inhibition OR inhibitor OR "volatile fatty acids" OR VFA OR ammonia OR lactate OR ethanol OR "process stability") (biohydrogen OR "hydrogen production")' \
"data/pdfs/openalex_q9_inhibition" \
"data/processed/openalex_q9_inhibition.jsonl"

run_one "q10_scaleup_pilot" \
'"dark fermentation" ("food waste" OR OFMSW) (pilot OR "pilot-scale" OR "demonstration" OR scale-up OR industrial) (biohydrogen OR "hydrogen production")' \
"data/pdfs/openalex_q10_scaleup" \
"data/processed/openalex_q10_scaleup.jsonl"

run_one "q11_TEA_LCA" \
'("dark fermentation" OR biohydrogen) ("food waste" OR OFMSW) ("techno-economic" OR TEA OR "life cycle assessment" OR LCA OR "carbon footprint" OR sustainability)' \
"data/pdfs/openalex_q11_tea_lca" \
"data/processed/openalex_q11_tea_lca.jsonl"

run_one "q12_integration_two_stage" \
'("dark fermentation" OR biohydrogen) ("food waste" OR OFMSW) ("photo fermentation" OR "microbial electrolysis cell" OR MEC OR "two-stage" OR "integrated process" OR "anaerobic digestion")' \
"data/pdfs/openalex_q12_integration" \
"data/processed/openalex_q12_integration.jsonl"

echo "[ALL DONE] q1~q12 finished."
