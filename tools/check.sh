#!/usr/bin/env bash
# check.sh — the full verification gate: format, lints, entire test
# suite (release mode, matching how experiments run). Run before every
# push; CI-equivalent for this repo.
#
# Usage: tools/check.sh [--quick]
#   --quick   skip clippy (fmt + tests only)
set -euo pipefail
cd "$(dirname "$0")/.."

QUICK=${1:-}
echo "== cargo fmt --check"
cargo fmt --check
if [ "$QUICK" != "--quick" ]; then
  echo "== cargo clippy (release, all targets)"
  cargo clippy --release --all-targets --quiet -- -D warnings
fi
echo "== cargo test (release)"
cargo test --release --quiet
echo "== OK"
