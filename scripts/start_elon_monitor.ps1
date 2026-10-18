param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pidPath = Join-Path $projectRoot "elon-monitor.pid"
$stdoutPath = Join-Path $projectRoot "elon-monitor.stdout.log"
$stderrPath = Join-Path $projectRoot "elon-monitor.stderr.log"
$executable = Join-Path $projectRoot ".venv\Scripts\elon-arb.exe"

if (-not $env:MYSQL_PASSWORD) {
    throw "Set MYSQL_PASSWORD in this PowerShell session before starting the monitor."
}

if (Test-Path -LiteralPath $pidPath) {
    $existingPid = [int](Get-Content -LiteralPath $pidPath -Raw)
    $existing = Get-Process -Id $existingPid -ErrorAction SilentlyContinue
    if ($existing) {
        throw "Elon monitor is already running with PID $existingPid."
    }
    Remove-Item -LiteralPath $pidPath
}

$process = Start-Process `
    -FilePath $executable `
    -ArgumentList "--observe-only" `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -WindowStyle Hidden `
    -PassThru

Set-Content -LiteralPath $pidPath -Value $process.Id -Encoding ascii
Write-Output "Elon observe-only monitor started with PID $($process.Id)."
