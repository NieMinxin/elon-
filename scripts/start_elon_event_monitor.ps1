param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-z0-9-]+$')]
    [string]$Name,

    [Parameter(Mandatory = $true)]
    [string]$EventSlug,

    [string]$Handle = "elonmusk"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeRoot = Join-Path $projectRoot "runtime\monitors"
$pidPath = Join-Path $runtimeRoot "$Name.pid"
$stdoutPath = Join-Path $runtimeRoot "$Name.stdout.log"
$stderrPath = Join-Path $runtimeRoot "$Name.stderr.log"
$statePath = Join-Path $runtimeRoot "$Name-state.json"
$auditPath = Join-Path $runtimeRoot "$Name-audit.jsonl"
$executable = Join-Path $projectRoot ".venv\Scripts\elon-arb.exe"
$gpuPython = Join-Path $projectRoot ".venv-gpu\Scripts\python.exe"
$originalRun = Join-Path $projectRoot "models\gru-original"
$v3Run = Join-Path $projectRoot "models\gru-v3"
$v4Run = Join-Path $projectRoot "models\gru-v4"

if (-not $env:MYSQL_PASSWORD) {
    throw "Set MYSQL_PASSWORD in this PowerShell session before starting the monitor."
}

New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null

$intervalConfig = Get-ChildItem -LiteralPath (Join-Path $projectRoot "config") -Filter "*.json" |
    Where-Object {
        (Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json).event_slug -eq $EventSlug
    } |
    Select-Object -First 1
if (-not $intervalConfig) {
    throw "No interval config matches event slug $EventSlug."
}
foreach ($requiredPath in @($gpuPython, $originalRun, $v3Run, $v4Run)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Four-model alert dependency is missing: $requiredPath"
    }
}

if (Test-Path -LiteralPath $pidPath) {
    $existingPid = [int](Get-Content -LiteralPath $pidPath -Raw)
    $existing = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
    if ($existing) {
        throw "Monitor $Name is already running with PID $existingPid."
    }
    Remove-Item -LiteralPath $pidPath
}

$env:ELON_EVENT_SLUG = $EventSlug
$env:ELON_XTRACKER_HANDLE = $Handle
$env:ELON_FORECAST_STRATEGY = "route"
$env:ELON_STATE_PATH = $statePath
$env:ELON_AUDIT_PATH = $auditPath
$env:ELON_ENABLE_LIVE = "false"
$env:ELON_POLL_INTERVAL_SECONDS = "120"
$env:ELON_FOUR_MODEL_ALERTS = "true"
$env:ELON_FOUR_MODEL_PYTHON = $gpuPython
$env:ELON_INTERVAL_CONFIG_PATH = $intervalConfig.FullName
$env:ELON_GRU_ORIGINAL_RUN = $originalRun
$env:ELON_GRU_V3_RUN = $v3Run
$env:ELON_GRU_V4_RUN = $v4Run
$env:ELON_GRU_DEVICE = "auto"
$env:ELON_FOUR_MODEL_OUTPUT_PATH = Join-Path $runtimeRoot "$Name-four-model.json"

$process = Start-Process `
    -FilePath $executable `
    -ArgumentList "--observe-only" `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -WindowStyle Hidden `
    -PassThru

Set-Content -LiteralPath $pidPath -Value $process.Id -Encoding ascii
Write-Output "Observe-only monitor $Name started with PID $($process.Id)."
