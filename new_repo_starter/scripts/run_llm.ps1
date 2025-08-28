# Resume-aware runner for build_essence_texts.py
# Usage:
#   ./new_repo_starter/scripts/run_llm.ps1 -Input "Tabulated Pivots ONLY_enriched_with_essence_v2_texts.xlsx" -Output "Tabulated Pivots ONLY_enhanced_llm_fresh.xlsx" -Log "llm_run.log"

param(
  [string]$Input = "Tabulated Pivots ONLY_enriched_with_essence_v2_texts.xlsx",
  [string]$Output = "Tabulated Pivots ONLY_enhanced_llm_fresh.xlsx",
  [string]$MakeCol = "Make",
  [string]$ModelCol = "Model",
  [string]$YearsCol = "Series (production years start-end)",
  [string]$BodyCol = "body type",
  [string]$LenCol = "Average of Length (mm)",
  [string]$EngineCol = "engine type",
  [string]$TransCol = "Transmission",
  [int]$ProgressEvery = 50,
  [int]$CheckpointEvery = 50,
  [string]$Log = "llm_run.log"
)

$ErrorActionPreference = "Stop"

$startIndex = 0
if (Test-Path processing_position.txt) {
  try { $startIndex = [int](Get-Content processing_position.txt) } catch { $startIndex = 0 }
}

$cmd = "python build_essence_texts.py --input `"$Input`" --output `"$Output`" --mode llm_variant --make-col `"$MakeCol`" --model-col `"$ModelCol`" --years-col `"$YearsCol`" --body-col `"$BodyCol`" --len-col `"$LenCol`" --engine-col `"$EngineCol`" --trans-col `"$TransCol`" --start-index $startIndex --progress-every $ProgressEvery --checkpoint-every $CheckpointEvery"

"==== $(Get-Date -Format s) starting ====" | Tee-Object -FilePath $Log -Append | Out-Null
"cmd: $cmd" | Tee-Object -FilePath $Log -Append | Out-Null

# Run and tee output to log
& powershell -NoProfile -Command $cmd 2>&1 | Tee-Object -FilePath $Log -Append

"==== $(Get-Date -Format s) finished ====" | Tee-Object -FilePath $Log -Append | Out-Null
