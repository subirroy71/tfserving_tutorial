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

# Upstream declares a Go package per .proto file (or none at all). The M flags
# below fold them into one Go package per directory instead
# ("import/path;package_name"), e.g. tensorflow_serving/apis/*.proto ->
# $MOD/gen/tensorflow_serving/apis, package apis.
FILES=()
M=()
while IFS= read -r f; do
  FILES+=("../proto/$f")
  dir=$(dirname "$f")
  case "$dir" in
    tensorflow/core/protobuf) pkg=coreprotobuf ;;  # avoid clashing with google's "protobuf"
    xla/tsl/protobuf) pkg=tslprotobuf ;;
    *) pkg=$(basename "$dir") ;;
  esac
  M+=("--go_opt=M$f=$MOD/gen/$dir;$pkg" "--go-grpc_opt=M$f=$MOD/gen/$dir;$pkg")
done < <(cd ../proto && find . -name '*.proto' | sed 's|^\./||' | sort)

rm -rf gen
$PROTOC -I ../proto \
  --go_out=. --go_opt=module=$MOD \
  --go-grpc_out=. --go-grpc_opt=module=$MOD \
  "${M[@]}" "${FILES[@]}"
echo "generated go/gen"
