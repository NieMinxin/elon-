#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <name>" >&2
  exit 2
fi

name="$1"
if [[ ! "$name" =~ ^[a-z0-9-]+$ ]]; then
  echo "Monitor name must contain only lowercase letters, digits, and hyphens." >&2
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pid_path="$project_root/runtime/monitors/$name.pid"
if [[ ! -f "$pid_path" ]]; then
  echo "Monitor $name PID file does not exist."
  exit 0
fi

monitor_pid="$(<"$pid_path")"
if kill -0 "$monitor_pid" 2>/dev/null; then
  kill "$monitor_pid"
  echo "Stopped monitor $name PID $monitor_pid."
else
  echo "Monitor $name PID $monitor_pid is not running."
fi
rm -f "$pid_path"
