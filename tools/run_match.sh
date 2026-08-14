#!/usr/bin/env bash
# Fastchess match under the fixed §27 protocol:
#   one thread per engine, ponder off, no tablebases, fixed hash,
#   shared opening suite applied symmetrically with colour-swapped
#   pairs, fixed random seed, all games saved, full log kept.
#
# Usage:
#   run_match.sh <name> <engine1-spec> <engine2-spec> <pairs> <limit> [seed]
# where <engine-spec> is one of:
#   ckpt:<checkpoint-dir>   our engine with a model
#   zero                    our engine, zero evaluator
#   random:<seed>           our engine, uniform random mover
#   sf-elo:<elo>            Stockfish with UCI_LimitStrength at <elo>
#   sf-nodes:<n>            Stockfish limited to <n> nodes per move
# and <limit> is either tc=<t>+<inc> or st=<sec> or nodes=<n>
# (applied to both engines; sf-nodes overrides its own node limit).
set -euo pipefail
cd "$(dirname "$0")/.."

NAME=$1; E1=$2; E2=$3; PAIRS=$4; LIMIT=$5; SEED=${6:-42}
FASTCHESS=${FASTCHESS:-$HOME/tools/fastchess/fastchess}
if [ ! -x "$FASTCHESS" ]; then
  echo "fastchess not found at $FASTCHESS — install it with:" >&2
  echo "  mkdir -p ~/tools/fastchess && cd ~/tools/fastchess && \\" >&2
  echo "  curl -sL https://github.com/Disservin/fastchess/releases/download/v1.8.2-alpha/fastchess-linux-x86-64.tar -o fc.tar && \\" >&2
  echo "  tar xf fc.tar && chmod +x fastchess-linux-x86-64/fastchess && ln -sf fastchess-linux-x86-64/fastchess fastchess" >&2
  exit 1
fi
UCI=./target/release/uci
SF=/usr/games/stockfish
OPENINGS=tools/openings_4ply.epd
OUT="runs/$(date -u +%Y%m%d-%H%M%S)-match-${NAME}-$(git rev-parse --short HEAD)"
mkdir -p "$OUT"

engine_args() {
  local spec=$1 label=$2
  case "$spec" in
    ckpt:*)   echo "-engine cmd=$UCI args=${spec#ckpt:} name=$label" ;;
    zero)     echo "-engine cmd=$UCI name=$label" ;;
    random:*) echo "-engine cmd=$UCI args=--random=${spec#random:} name=$label" ;;
    sf-elo:*) echo "-engine cmd=$SF name=$label option.UCI_LimitStrength=true option.UCI_Elo=${spec#sf-elo:} option.Threads=1 option.Hash=16" ;;
    sf-nodes:*) echo "-engine cmd=$SF name=$label option.Threads=1 option.Hash=16 nodes=${spec#sf-nodes:}" ;;
    *) echo "unknown engine spec $spec" >&2; exit 1 ;;
  esac
}

LIMIT_ARG=""
case "$LIMIT" in
  tc=*)    LIMIT_ARG="tc=${LIMIT#tc=}" ;;
  st=*)    LIMIT_ARG="st=${LIMIT#st=}" ;;
  nodes=*) LIMIT_ARG="tc=inf nodes=${LIMIT#nodes=}" ;;
  *) echo "unknown limit $LIMIT" >&2; exit 1 ;;
esac

{
  echo "name=$NAME e1=$E1 e2=$E2 pairs=$PAIRS limit=$LIMIT seed=$SEED"
  echo "openings=$(sha256sum $OPENINGS)"
  echo "uci=$(sha256sum $UCI | cut -d' ' -f1) stockfish=$($SF --help </dev/null 2>&1 | head -1 || true)"
} > "$OUT/protocol.txt"

# shellcheck disable=SC2046
"$FASTCHESS" \
  $(engine_args "$E1" A) \
  $(engine_args "$E2" B) \
  -each $LIMIT_ARG timemargin=${TIMEMARGIN:-100} \
  -openings file=$OPENINGS format=epd order=random \
  -rounds "$PAIRS" -games 2 -repeat \
  -srand "$SEED" \
  -concurrency "${CONCURRENCY:-3}" \
  -pgnout file="$OUT/games.pgn" \
  2>&1 | tee "$OUT/fastchess.log" | tail -12
