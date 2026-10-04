#!/usr/bin/env bash
# Generate the Python protobuf/gRPC modules into python/gen/.
set -euo pipefail
# Resolve a relative interpreter path (PYTHON=.venv/bin/python) before cd.
PY="${PYTHON:-python3}"
case "$PY" in */*) PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")" ;; esac
cd "$(dirname "$0")"
rm -rf gen && mkdir -p gen
# shellcheck disable=SC2046
"$PY" -m grpc_tools.protoc -I ../proto \
  --python_out=gen --grpc_python_out=gen \
  $(cd ../proto && find . -name '*.proto' | sed 's|^\./|../proto/|' | sort)
# Regular packages (not namespace packages), so these stubs win over an
# installed TensorFlow when python/gen is first on sys.path.
find gen -type d -exec touch {}/__init__.py \;
echo "generated python/gen"
