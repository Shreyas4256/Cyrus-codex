#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install -r requirements-ml.lock \
  --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/cyrus doctor --config configs/smoke.yaml

