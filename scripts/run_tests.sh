#!/usr/bin/env bash
# Run the test suite from the repo root (pytest.ini supplies pythonpath=. and testpaths=tests).
set -euo pipefail
cd "$(dirname "$0")/.."
exec python -m pytest tests/ -q "$@"
