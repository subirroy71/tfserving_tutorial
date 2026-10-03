#!/usr/bin/env bash
# Regenerate go/gen/ from ../proto. The generated code is committed, so this
# is only needed after changing the .proto files.
#
# Needs: protoc (or `pip install grpcio-tools`, which bundles one) and
#   go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
#   go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$PATH:$(go env GOPATH)/bin"

PROTOC=protoc
command -v protoc >/dev/null || PROTOC="python3 -m grpc_tools.protoc"

MOD=github.com/subirroy71/tfserving_tutorial/go
FW=$MOD/gen/tensorflow/core/framework
API=$MOD/gen/tensorflow_serving/apis

# The upstream files declare one Go package per .proto (or none at all); the
# M flags below fold them into two packages inside this module instead
# ("import/path;package_name").
M=()
for f in tensor tensor_shape types resource_handle; do
  M+=("--go_opt=Mtensorflow/core/framework/$f.proto=$FW;framework" "--go-grpc_opt=Mtensorflow/core/framework/$f.proto=$FW;framework")
done
for f in model predict prediction_service; do
  M+=("--go_opt=Mtensorflow_serving/apis/$f.proto=$API;apis" "--go-grpc_opt=Mtensorflow_serving/apis/$f.proto=$API;apis")
done

rm -rf gen
$PROTOC -I ../proto \
  --go_out=. --go_opt=module=$MOD \
  --go-grpc_out=. --go-grpc_opt=module=$MOD \
  "${M[@]}" \
  ../proto/tensorflow/core/framework/*.proto \
  ../proto/tensorflow_serving/apis/*.proto
echo "generated go/gen"
