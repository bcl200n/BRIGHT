#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m src.infer --config config/disaster.yaml --ann-file data/instance_annotations/val.json "$@"
