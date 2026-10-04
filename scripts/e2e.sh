#!/usr/bin/env bash
# Run the Python, Go and Rust clients against a running model server (started
# with `docker compose up`) and check that, for every case below, all three
# exit with the expected status and print the same JSON.
#
#   ./scripts/e2e.sh [host:port]      (default localhost:8500)
#
# PYTHON and CARGO override the interpreter / cargo binary used.
set -euo pipefail
cd "$(dirname "$0")/.."
ADDR="${1:-localhost:8500}"
PYTHON="${PYTHON:-python3}"
CARGO="${CARGO:-cargo}"

[ -d python/gen ] || ./python/gen_protos.sh >/dev/null
(cd go && go build -o bin/client ./cmd/client)
(cd rust && "$CARGO" build --quiet)

# expected exit status | client arguments
CASES=(
  "0|predict --input examples/applicants.json"
  "0|predict --version 1 --input examples/applicants.json"
  "0|predict --label canary --input examples/applicants.json"
  "0|predict --signature half_plus_two --input examples/half_plus_two_matrix.json"
  "1|predict --signature half_plus_two --input examples/half_plus_two_scalar.json"
  "1|predict --label nope --signature half_plus_two --input examples/half_plus_two_matrix.json"
  "0|predict --model iris --signature predict --input examples/iris_predict.json"
  "0|classify --model iris --input examples/iris_classify.json"
  "0|regress --model iris --signature regress --input examples/iris_regress.json"
  "1|classify --model iris --signature regress --input examples/iris_classify.json"
  "0|metadata --model demo"
  "0|metadata --model iris"
  "0|status --model demo"
  "0|status --model iris --version 1"
  "2|status --model demo --label stable"
)

run() {  # run <name> <command...>: sets out_<name> and rc_<name>
  local name=$1; shift
  local out rc=0
  out=$("$@" 2>/dev/null) || rc=$?
  printf -v "out_$name" '%s' "$out"
  printf -v "rc_$name" '%s' "$rc"
}

fail=0
for case in "${CASES[@]}"; do
  want=${case%%|*}
  read -r -a args <<<"${case#*|}"
  args+=(--addr "$ADDR")

  run py "$PYTHON" python/client.py "${args[@]}"
  run go go/bin/client "${args[@]}"
  run rs rust/target/debug/client "${args[@]}"

  label="${case#*|}"
  if [ "$rc_py$rc_go$rc_rs" != "$want$want$want" ]; then
    echo "FAIL $label: exit status python=$rc_py go=$rc_go rust=$rc_rs, want $want"
    fail=1
    continue
  fi
  # Compare as parsed JSON, so number formatting (2 vs 2.0) does not matter.
  if [ "$want" != 0 ] || "$PYTHON" - "$out_py" "$out_go" "$out_rs" <<'PY'
import json, sys
py, go, rs = (json.loads(a) for a in sys.argv[1:4])
sys.exit(0 if py == go == rs else 1)
PY
  then
    echo "ok   $label"
  else
    echo "FAIL $label: clients disagree"
    printf 'python: %s\ngo:     %s\nrust:   %s\n' "$out_py" "$out_go" "$out_rs"
    fail=1
  fi
done
exit $fail
