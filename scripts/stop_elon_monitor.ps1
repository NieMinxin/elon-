param()

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pidPath = Join-Path $projectRoot "elon-monitor.pid"

if (-not (Test-Path -LiteralPath $pidPath)) {
    Write-Output "Elon monitor PID file does not exist."
    exit 0
}

$monitorPid = [int](Get-Content -LiteralPath $pidPath -Raw)
$process = Get-Process -Id $monitorPid -ErrorAction SilentlyContinue
if ($process) {
    Stop-Process -Id $monitorPid
    Write-Output "Stopped Elon monitor PID $monitorPid."
} else {
    Write-Output "Elon monitor PID $monitorPid is not running."
}
Remove-Item -LiteralPath $pidPath
