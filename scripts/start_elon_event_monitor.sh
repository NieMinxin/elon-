#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <name> <event-slug> [xtracker-handle]" >&2
  exit 2
fi

name="$1"
event_slug="$2"
handle="${3:-elonmusk}"

if [[ ! "$name" =~ ^[a-z0-9-]+$ ]]; then
  echo "Monitor name must contain only lowercase letters, digits, and hyphens." >&2
  exit 2
fi
if [[ -z "${MYSQL_PASSWORD:-}" ]]; then
  echo "Set MYSQL_PASSWORD before starting the monitor." >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="$project_root/runtime/monitors"
pid_path="$runtime_root/$name.pid"
python_path="$project_root/.venv/bin/python"
executable="$project_root/.venv/bin/elon-arb"

if [[ ! -x "$python_path" || ! -x "$executable" ]]; then
  echo "Create .venv and install the project with .[dev,ml] first." >&2
  exit 1
fi

interval_config="$($python_path -c '
import json, pathlib, sys
matches = [p for p in pathlib.Path(sys.argv[1]).glob("*.json")
           if json.loads(p.read_text(encoding="utf-8")).get("event_slug") == sys.argv[2]]
if len(matches) != 1:
    raise SystemExit(1)
print(matches[0].resolve())
' "$project_root/config" "$event_slug")" || {
  echo "No unique interval config matches event slug $event_slug." >&2
  exit 1
}

for required in \
  "$project_root/models/gru-original" \
  "$project_root/models/gru-v3" \
  "$project_root/models/gru-v4"; do
  if [[ ! -e "$required" ]]; then
    echo "Four-model alert dependency is missing: $required" >&2
    exit 1
  fi
done

mkdir -p "$runtime_root"
if [[ -f "$pid_path" ]]; then
  existing_pid="$(<"$pid_path")"
  if kill -0 "$existing_pid" 2>/dev/null; then
    echo "Monitor $name is already running with PID $existing_pid." >&2
    exit 1
  fi
  rm -f "$pid_path"
fi

export ELON_EVENT_SLUG="$event_slug"
export ELON_XTRACKER_HANDLE="$handle"
export ELON_FORECAST_STRATEGY="route"
export ELON_STATE_PATH="$runtime_root/$name-state.json"
export ELON_AUDIT_PATH="$runtime_root/$name-audit.jsonl"
export ELON_ENABLE_LIVE="false"
export ELON_POLL_INTERVAL_SECONDS="120"
export ELON_FOUR_MODEL_ALERTS="true"
export ELON_FOUR_MODEL_PYTHON="$python_path"
export ELON_INTERVAL_CONFIG_PATH="$interval_config"
export ELON_GRU_ORIGINAL_RUN="$project_root/models/gru-original"
export ELON_GRU_V3_RUN="$project_root/models/gru-v3"
export ELON_GRU_V4_RUN="$project_root/models/gru-v4"
export ELON_GRU_DEVICE="auto"
export ELON_FOUR_MODEL_OUTPUT_PATH="$runtime_root/$name-four-model.json"

nohup "$executable" --observe-only \
  >"$runtime_root/$name.stdout.log" \
  2>"$runtime_root/$name.stderr.log" &
monitor_pid=$!
echo "$monitor_pid" >"$pid_path"
echo "Observe-only monitor $name started with PID $monitor_pid."
