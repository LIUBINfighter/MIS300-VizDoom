#!/usr/bin/env bash
set -euo pipefail
METHOD=${1:-pip}
VENV=${2:-.venv}

echo "Install method: $METHOD"
if [ "$METHOD" = "pip" ]; then
  if [ ! -d "$VENV" ]; then
    python -m venv "$VENV"
  fi
  # shellcheck source=/dev/null
  source "$VENV/bin/activate"
  if [ -f requirements.txt ]; then
    python -m pip install --upgrade pip
    pip install -r requirements.txt
  else
    echo "requirements.txt not found. Trying to generate from Poetry..."
    if command -v poetry >/dev/null 2>&1; then
      poetry export -f requirements.txt --without-hashes -o requirements.txt
      python -m pip install --upgrade pip
      pip install -r requirements.txt
    else
      echo "requirements.txt missing and poetry not available. Install poetry or add requirements.txt." >&2
      exit 1
    fi
  fi
else
  if command -v poetry >/dev/null 2>&1; then
    poetry install
  else
    echo "poetry not found. Install poetry or use method pip." >&2
    exit 1
  fi
fi

echo "Install completed."