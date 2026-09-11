#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH="$PWD:$PWD/alphaapollo/core/generation" python -m pytest tests/harness "$@"
