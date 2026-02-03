#!/usr/bin/env bash
set -euo pipefail

MAX_RESULTS=2000
POLITE_DELAY=1.0
TIMEOUT=60

declare -A QUERIES

QUERIES[q1_core]='"dark fermentation" (biohydrogen OR "hydrogen production") ("food waste" OR "kitchen waste" OR OFMSW OR "organic fraction") (pH OR temperature OR HRT OR OLR OR "C/N" OR inoculum OR pretreatment OR "hydrogen yield")'
QUERIES[q2_pretreatment]='"dark fermentation" ("food waste" OR "kitchen waste" OR OFMSW OR "organic fraction") (pretreatment OR "thermal pretreatment" OR "acid pretreatment" OR alkaline OR enzymatic OR ultrasound OR microwave OR "steam explosion") (biohydrogen OR "hydrogen production" OR "H2 production")'
QUERIES[q3_inoculum]='"dark fermentation" ("food waste" OR "kitchen waste" OR OFMSW OR "organic fraction") (inoculum OR "anaerobic sludge" OR "heat shock" OR "acid treatment" OR "inoculum pretreatment" OR methanogen) (biohydrogen OR "hydrogen production")'
QUERIES[q4_operating]='"dark fermentation" (biohydrogen OR "hydrogen production") ("food waste" OR "kitchen waste" OR OFMSW OR "organic fraction") (pH OR temperature OR HRT OR OLR OR "C/N" OR agitation OR ORP OR "hydrogen partial pressure")'
QUERIES[q5_reactor]='"dark fermentation" ("food waste" OR OFMSW OR "organic fraction") (CSTR OR UASB OR "packed bed" OR "fluidized bed" OR "membrane bioreactor" OR "sequencing batch" OR continuous OR "two-stage") (biohydrogen OR "hydrogen production")'
QUERIES[q6_codigestion]='("dark fermentation" OR "anaerobic fermentation") (biohydrogen OR "hydrogen production") ("food waste" OR OFMSW) ("co-digestion" OR "co-fermentation" OR "co-substrate")'
QUERIES[q7_microbiome]='"dark fermentation" ("food waste" OR OFMSW OR "organic fraction") (Clostridium OR Enterobacter OR "microbial community" OR metagenomics OR "16S rRNA" OR pathway OR "butyrate-type" OR "acetate-type") (biohydrogen OR "hydrogen production")'
QUERIES[q8_kinetics]='("dark fermentation" OR biohydrogen) ("food waste" OR OFMSW) (kinetic OR kinetics OR "modified Gompertz" OR Gompertz OR "first-order" OR "hydrogen production rate")'
QUERIES[q9_inhibition]='"dark fermentation" ("food waste" OR OFMSW) (inhibition OR inhibitor OR "volatile fatty acids" OR VFA OR ammonia OR lactate OR ethanol OR "process stability") (biohydrogen OR "hydrogen production")'
QUERIES[q10_scaleup]='"dark fermentation" ("food waste" OR OFMSW) (pilot OR "pilot-scale" OR "demonstration" OR scale-up OR industrial) (biohydrogen OR "hydrogen production")'
QUERIES[q11_tea_lca]='("dark fermentation" OR biohydrogen) ("food waste" OR OFMSW) ("techno-economic" OR TEA OR "life cycle assessment" OR LCA OR "carbon footprint" OR sustainability)'
QUERIES[q12_integration]='("dark fermentation" OR biohydrogen) ("food waste" OR OFMSW) ("photo fermentation" OR "microbial electrolysis cell" OR MEC OR "two-stage" OR "integrated process" OR "anaerobic digestion")'

for name in "${!QUERIES[@]}"; do
  out_dir="data/pdfs/openalex_${name}"
  meta_out="data/processed/openalex_${name}.jsonl"
  echo "[RUN] ${name}"
  conda run -n envgpt python -m src.data.oa_pdf_crawler \
    --mode openalex \
    --query "${QUERIES[$name]}" \
    --max_results "${MAX_RESULTS}" \
    --out_dir "${out_dir}" \
    --meta_out "${meta_out}" \
    --polite_delay "${POLITE_DELAY}" \
    --timeout "${TIMEOUT}"
done
