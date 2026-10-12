$ErrorActionPreference = "Stop"

$ollama = "D:\software\Ollama\ollama.exe"
if (-not (Test-Path $ollama)) {
  $ollama = (Get-Command ollama -ErrorAction Stop).Source
}

# The models already installed on this machine are in the default user model
# directory. Bind only to localhost; Docker communication is handled by the
# PowerShell bridge in run_one_ollama.ps1.
$env:OLLAMA_MODELS = Join-Path $env:USERPROFILE ".ollama\models"
$env:OLLAMA_HOST = "127.0.0.1:11434"

Write-Host "Starting Ollama on $env:OLLAMA_HOST"
Write-Host "Model directory: $env:OLLAMA_MODELS"
Write-Host "Keep this window open while run_one_ollama.ps1 is running."
& $ollama serve
