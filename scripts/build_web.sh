#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out_dir="${1:-"$repo_dir/web"}"
mkdir -p "$out_dir"

GOOS=js GOARCH=wasm go build -o "$out_dir/mirage.wasm" ./cmd/mirage-wasm

wasm_exec="$(go env GOROOT)/lib/wasm/wasm_exec.js"
if [[ ! -f "$wasm_exec" ]]; then
  wasm_exec="$(go env GOROOT)/misc/wasm/wasm_exec.js"
fi
cp "$wasm_exec" "$out_dir/wasm_exec.js"

echo "web runtime written to $out_dir"
