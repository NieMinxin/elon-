param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-z0-9-]+$')]
    [string]$Name
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pidPath = Join-Path $projectRoot "runtime\monitors\$Name.pid"

if (-not (Test-Path -LiteralPath $pidPath)) {
    Write-Output "Monitor $Name PID file does not exist."
    exit 0
}

$monitorPid = [int](Get-Content -LiteralPath $pidPath -Raw)
$process = Get-Process -Id $monitorPid -ErrorAction SilentlyContinue
if ($process) {
    Stop-Process -Id $monitorPid
    Write-Output "Stopped monitor $Name PID $monitorPid."
} else {
    Write-Output "Monitor $Name PID $monitorPid is not running."
}
Remove-Item -LiteralPath $pidPath
