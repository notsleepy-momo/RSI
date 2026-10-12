param(
  [string]$Model = "qwen3.5:4b",
  [int]$TaskIndex = 0,
  [int]$MaxSteps = 1,
  [string]$Split = "train",
  [switch]$DirectContainerOllama
)

$ErrorActionPreference = "Stop"
$project = (Resolve-Path $PSScriptRoot).Path
$data = Join-Path $project "data"
$results = Join-Path $project "results"
New-Item -ItemType Directory -Force -Path $data, $results | Out-Null

$docker = "D:\software\Docker\resources\bin\docker.exe"
if (-not (Test-Path $docker)) { $docker = "docker" }

try {
  $tags = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 10
} catch {
  throw "Ollama is not reachable on localhost. Open another PowerShell window and run .\start_ollama_localhost.ps1"
}
if (-not ($tags.models.name -contains $Model)) {
  throw "Ollama model '$Model' is not visible. On this machine the existing models are under $env:USERPROFILE\.ollama\models; run .\start_ollama_localhost.ps1"
}

Write-Host "[1/3] Building ALFWorld text/Ollama image..."
& $docker build --file (Join-Path $project "Dockerfile.ollama") --tag alfworld-ollama:qwen35 $project

Write-Host "[2/3] Downloading official ALFWorld text data (one-time, about 145 MB)..."
& $docker run --rm `
  --mount "type=bind,source=$data,target=/opt/alfworld_data" `
  --entrypoint python alfworld-ollama:qwen35 `
  /opt/alfworld_project/download_alfworld_data.py --data-dir /opt/alfworld_data

$out = "/results/alfworld_${Split}_${TaskIndex}_qwen35_4b.json"
if ($DirectContainerOllama) {
  Write-Host "[3/3] Running one task with container -> host.docker.internal Ollama..."
  & $docker run --rm `
    --mount "type=bind,source=$data,target=/opt/alfworld_data" `
    --mount "type=bind,source=$results,target=/results" `
    alfworld-ollama:qwen35 `
    --model $Model --split $Split --task-index $TaskIndex --max-steps $MaxSteps --output $out
} else {
  if ($MaxSteps -ne 1) { throw "localhost bridge mode supports MaxSteps=1; use -DirectContainerOllama for multiple steps" }
  Write-Host "[3a/3] Preparing one official ALFWorld observation/action list..."
  & $docker run --rm `
    --mount "type=bind,source=$data,target=/opt/alfworld_data" `
    --mount "type=bind,source=$results,target=/results" `
    alfworld-ollama:qwen35 `
    --model $Model --split $Split --task-index $TaskIndex --max-steps 1 `
    --prepare-only --output /results/alfworld_request.json

  Write-Host "[3b/3] Calling Ollama on Windows localhost..."
  $requestPath = Join-Path $results "alfworld_request.json"
  $responsePath = Join-Path $results "ollama_model_response.json"
  $request = Get-Content -Raw -Encoding UTF8 $requestPath | ConvertFrom-Json
  $body = $request.ollama_payload | ConvertTo-Json -Depth 20 -Compress
  $ollama = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:11434/api/chat" `
    -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 240
  try {
    $modelObject = $ollama.message.content | ConvertFrom-Json -ErrorAction Stop
  } catch {
    # A small local model can hit num_predict after already emitting a valid
    # action field. Preserve that field instead of discarding the whole run.
    $match = [regex]::Match($ollama.message.content, '"action"\s*:\s*"([^"]+)"')
    if (-not $match.Success) { throw }
    $modelObject = [pscustomobject]@{
      action = $match.Groups[1].Value
      raw_truncated_output = $ollama.message.content
    }
  }
  $modelObject | ConvertTo-Json -Depth 10 | Set-Content -Encoding UTF8 $responsePath

  Write-Host "[3c/3] Replaying the model action in the same official task..."
  & $docker run --rm `
    --mount "type=bind,source=$data,target=/opt/alfworld_data" `
    --mount "type=bind,source=$results,target=/results" `
    alfworld-ollama:qwen35 `
    --model $Model --split $Split --task-index $TaskIndex --max-steps 1 `
    --ollama-url http://127.0.0.1:11434 `
    --model-response-input /results/ollama_model_response.json --output $out
}

Write-Host "Result: $(Join-Path $results (Split-Path $out -Leaf))"
