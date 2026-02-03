$jobs = @(
  @{ name="q1_core"; q='"dark fermentation" (biohydrogen OR "hydrogen production") ("food waste" OR "kitchen waste" OR OFMSW OR "organic fraction") (pH OR temperature OR HRT OR OLR OR "C/N" OR inoculum OR pretreatment OR "hydrogen yield")' },
  @{ name="q2_pretreatment"; q='"dark fermentation" ("food waste" OR "kitchen waste" OR OFMSW OR "organic fraction") (pretreatment OR "thermal pretreatment" OR "acid pretreatment" OR alkaline OR enzymatic OR ultrasound OR microwave OR "steam explosion") (biohydrogen OR "hydrogen production" OR "H2 production")' },
  @{ name="q3_inoculum"; q='"dark fermentation" ("food waste" OR "kitchen waste" OR OFMSW OR "organic fraction") (inoculum OR "anaerobic sludge" OR "heat shock" OR "acid treatment" OR "inoculum pretreatment" OR methanogen) (biohydrogen OR "hydrogen production")' },
  @{ name="q4_operating"; q='"dark fermentation" (biohydrogen OR "hydrogen production") ("food waste" OR "kitchen waste" OR OFMSW OR "organic fraction") (pH OR temperature OR HRT OR OLR OR "C/N" OR agitation OR ORP OR "hydrogen partial pressure")' },
  @{ name="q5_reactor"; q='"dark fermentation" ("food waste" OR OFMSW OR "organic fraction") (CSTR OR UASB OR "packed bed" OR "fluidized bed" OR "membrane bioreactor" OR "sequencing batch" OR continuous OR "two-stage") (biohydrogen OR "hydrogen production")' },
  @{ name="q6_codigestion"; q='("dark fermentation" OR "anaerobic fermentation") (biohydrogen OR "hydrogen production") ("food waste" OR OFMSW) ("co-digestion" OR "co-fermentation" OR "co-substrate")' },
  @{ name="q7_microbiome"; q='"dark fermentation" ("food waste" OR OFMSW OR "organic fraction") (Clostridium OR Enterobacter OR "microbial community" OR metagenomics OR "16S rRNA" OR pathway OR "butyrate-type" OR "acetate-type") (biohydrogen OR "hydrogen production")' },
  @{ name="q8_kinetics"; q='("dark fermentation" OR biohydrogen) ("food waste" OR OFMSW) (kinetic OR kinetics OR "modified Gompertz" OR Gompertz OR "first-order" OR "hydrogen production rate")' },
  @{ name="q9_inhibition"; q='"dark fermentation" ("food waste" OR OFMSW) (inhibition OR inhibitor OR "volatile fatty acids" OR VFA OR ammonia OR lactate OR ethanol OR "process stability") (biohydrogen OR "hydrogen production")' },
  @{ name="q10_scaleup"; q='"dark fermentation" ("food waste" OR OFMSW) (pilot OR "pilot-scale" OR "demonstration" OR scale-up OR industrial) (biohydrogen OR "hydrogen production")' },
  @{ name="q11_tea_lca"; q='("dark fermentation" OR biohydrogen) ("food waste" OR OFMSW) ("techno-economic" OR TEA OR "life cycle assessment" OR LCA OR "carbon footprint" OR sustainability)' },
  @{ name="q12_integration"; q='("dark fermentation" OR biohydrogen) ("food waste" OR OFMSW) ("photo fermentation" OR "microbial electrolysis cell" OR MEC OR "two-stage" OR "integrated process" OR "anaerobic digestion")' }
)

foreach ($j in $jobs) {
  $outDir = "data/pdfs/openalex_$($j.name)"
  $metaOut = "data/processed/openalex_$($j.name).jsonl"
  conda run -n envgpt python -m src.data.oa_pdf_crawler `
    --mode openalex `
    --query $j.q `
    --max_results 2000 `
    --out_dir $outDir `
    --meta_out $metaOut `
    --polite_delay 1.0 `
    --timeout 60
}
