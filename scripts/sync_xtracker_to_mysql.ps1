[CmdletBinding()]
param(
    [string]$StartDate,
    [ValidateRange(0, 8760)]
    [int]$OverlapHours = 0,
    [string]$Handle = "elonmusk"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$executable = Join-Path $projectRoot ".venv\Scripts\elon-data.exe"

if (-not (Test-Path -LiteralPath $executable)) {
    throw "elon-data.exe was not found. Install the project dependencies first."
}
if (-not $env:MYSQL_PASSWORD) {
    throw "MYSQL_PASSWORD must be set in the current environment."
}

$arguments = @("--handle", $Handle, "sync", "--overlap-hours", $OverlapHours)
if ($StartDate) {
    $arguments += @("--start", $StartDate)
}

Push-Location $projectRoot
try {
    & $executable @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "xTracker incremental sync failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
