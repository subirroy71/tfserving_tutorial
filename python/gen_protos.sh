#!/usr/bin/env bash
# Generate the Python protobuf/gRPC modules into python/gen/.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf gen && mkdir -p gen
python3 -m grpc_tools.protoc -I ../proto \
  --python_out=gen --grpc_python_out=gen \
  ../proto/tensorflow/core/framework/*.proto \
  ../proto/tensorflow_serving/apis/*.proto
# Regular packages (not namespace packages), so these stubs win over an
# installed TensorFlow when python/gen is first on sys.path.
find gen -type d -exec touch {}/__init__.py \;
echo "generated python/gen"
