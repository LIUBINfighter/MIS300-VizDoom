#!/usr/bin/env bash
set -euo pipefail
if ! command -v poetry >/dev/null 2>&1; then
  echo "poetry is not installed; install it first to export requirements." >&2
  exit 1
fi

echo "Exporting requirements.txt from Poetry lock..."
poetry export -f requirements.txt --without-hashes -o requirements.txt
echo "Exported requirements.txt"