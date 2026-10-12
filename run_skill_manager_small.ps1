param(
  [string]$Model = "qwen3.5:4b",
  [int]$Rounds = 3,
  [int]$Patience = 2,
  [int]$MaxSteps = 20,
  [int]$ActionRetries = 2,
  [int]$StagnationWindow = 8,
  [double]$MinEfficiencyGain = 0.10,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$project = (Resolve-Path $PSScriptRoot).Path
$data = Join-Path $project "data"
$results = Join-Path $project "results"
New-Item -ItemType Directory -Force -Path $data, $results | Out-Null
$docker = "D:\software\Docker\resources\bin\docker.exe"
if (-not (Test-Path $docker)) { $docker = "docker" }

Write-Host "[1/3] Building the ALFWorld image..."
& $docker build --file (Join-Path $project "Dockerfile.ollama") --tag alfworld-ollama:qwen35 $project
if ($LASTEXITCODE -ne 0) { throw "Docker image build failed" }

if (-not $DryRun) {
  try {
    $tags = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 10
  } catch {
    throw "Ollama is not reachable. Start it with .\start_ollama_localhost.ps1"
  }
  if (-not ($tags.models.name -contains $Model)) { throw "Ollama model '$Model' is unavailable" }
  Write-Host "[2/3] Ensuring official ALFWorld data are present..."
  & $docker run --rm `
    --mount "type=bind,source=$data,target=/opt/alfworld_data" `
    --entrypoint python alfworld-ollama:qwen35 `
    /opt/alfworld_project/download_alfworld_data.py --data-dir /opt/alfworld_data
  if ($LASTEXITCODE -ne 0) { throw "ALFWorld data preparation failed" }
} else {
  Write-Host "[2/3] Dry run: skipping Ollama and ALFWorld data checks."
}

$runArgs = @(
  "run", "--rm",
  "--mount", "type=bind,source=$data,target=/opt/alfworld_data",
  "--mount", "type=bind,source=$results,target=/results",
  "--entrypoint", "python",
  "alfworld-ollama:qwen35",
  "/opt/alfworld_project/ollama_adapter/skill_manager_experiment.py",
  "--model", $Model,
  "--split-manifest", "/opt/alfworld_project/configs/splits_small_5.json",
  "--rounds", "$Rounds",
  "--patience", "$Patience",
  "--max-steps", "$MaxSteps",
  "--action-retries", "$ActionRetries",
  "--stagnation-window", "$StagnationWindow",
  "--min-efficiency-gain", "$MinEfficiencyGain",
  "--output-dir", "/results/skill_manager_small"
)
if ($DryRun) { $runArgs += "--dry-run" }

Write-Host "[3/3] Running multi-round Skill Manager experiment..."
& $docker @runArgs
if ($LASTEXITCODE -ne 0) { throw "Skill Manager experiment failed" }
Write-Host "Results: $(Join-Path $results 'skill_manager_small')"
