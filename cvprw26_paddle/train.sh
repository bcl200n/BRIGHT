#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m src.train --config config/disaster.yaml "$@"
