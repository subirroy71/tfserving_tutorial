#!/usr/bin/env bash
# Run the Python, Go and Rust clients against a running server and check that
# all three print the same outputs for every example request.
#
#   ./scripts/e2e.sh [host:port]      (default localhost:8500)
set -euo pipefail
cd "$(dirname "$0")/.."
ADDR="${1:-localhost:8500}"

[ -d python/gen ] || ./python/gen_protos.sh >/dev/null
(cd go && go build -o bin/client ./cmd/client)
(cd rust && cargo build --quiet)

fail=0
for request in examples/*.json; do
  signature=serving_default
  case "$request" in *half_plus_two*) signature=half_plus_two ;; esac
  args=(--addr "$ADDR" --model demo --signature "$signature" --input "$request")

  py=$(python3 python/client.py "${args[@]}" 2>/dev/null) || { echo "FAIL $request (python)"; fail=1; continue; }
  go=$(go/bin/client "${args[@]}" 2>/dev/null) || { echo "FAIL $request (go)"; fail=1; continue; }
  rs=$(rust/target/debug/client "${args[@]}" 2>/dev/null) || { echo "FAIL $request (rust)"; fail=1; continue; }

  # Compare as parsed JSON, so number formatting (2 vs 2.0) does not matter.
  if python3 - "$py" "$go" "$rs" <<'PY'
import json, sys
py, go, rs = (json.loads(a) for a in sys.argv[1:4])
sys.exit(0 if py == go == rs else 1)
PY
  then
    echo "ok   $request  $(echo "$py" | tr -d ' \n')"
  else
    echo "FAIL $request: clients disagree"; echo "python: $py"; echo "go: $go"; echo "rust: $rs"
    fail=1
  fi
done
exit $fail
