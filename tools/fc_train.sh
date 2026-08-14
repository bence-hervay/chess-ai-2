#!/usr/bin/env bash
# fc_train.sh — chunked, resumable Expert Iteration campaign driver.
#
# Runs `lab selfplay` in chunks of generations. Each chunk starts from
# the previous chunk's champion (init_checkpoint), so a campaign can be
# interrupted at any time and resumed with the same command; a chunk
# that was killed mid-run is simply re-run from its start (chunks are
# the atomic unit; an unfinished chunk leaves no DONE marker).
#
# Everything is deterministic given the campaign parameters: chunk c
# uses seed = base_seed + c - 1, and all lab internals are
# thread-count-independent. The FIFO replay window persists across
# chunks (replay.jsonl + init_replay), so a chunked campaign trains
# like one long run; per-chunk "vs gen0" progression is measured
# against the chunk's own starting champion.
#
# Layout under campaigns/<name>/:
#   campaign.env            frozen parameters (sourced on resume)
#   log.txt                 one line per chunk + streamed lab output
#   baseline_gen0/          the campaign's random-init checkpoint (Elo anchor)
#   chunk_001/ ... chunk_N/ per-chunk snapshot:
#       config.toml resolved.toml summary.json metrics.jsonl
#       checkpoint/         champion at the end of the chunk
#       run_dir.txt DONE
#   champion/               copy of the latest completed chunk's checkpoint
#
# Monitor a running campaign with:
#   tail -f campaigns/<name>/log.txt
# Rate the snapshots (approximate Elo curve) with:
#   tools/fc_rating.py --campaign campaigns/<name>
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
  cat <<'EOF'
Usage: tools/fc_train.sh <campaign-name> [options]

Creates or resumes the chunked self-play campaign campaigns/<name>.
On resume, saved parameters are used; new options are ignored, except
--chunks N which extends the target — that is also how you CONTINUE a
finished campaign with the same recipe (nothing retrains; the next
chunk picks up the champion and its replay window).

To continue a trained champion under a DIFFERENT recipe (deeper
search, more games, ...), fork it into a new campaign:

  tools/fc_train.sh <new-name> --from campaigns/<old> --nodes 800

--from accepts a campaign directory (champion + latest replay), a
chunk directory (that snapshot + its replay), or a bare checkpoint
directory (no replay). The new campaign's chunk 1 initializes from it;
width and game must match the source model. Each campaign's recipe
stays frozen, so provenance is never mixed; the fork's baseline_gen0
anchor is the source champion (its Elo curve reads "gain since fork").

Options (defaults in brackets):
  --from PATH       initialize chunk 1 from an existing campaign /
                    chunk / checkpoint (creation only) [none]
  --game G          fc-tiny | fc-small | fc-medium | fc-full [fc-full]
  --width W         model width [64]
  --chunks N        total chunks to reach (resume raises this) [6]
  --gens-per-chunk N  generations per chunk [8]
  --games N         self-play games per generation [200]
  --nodes N         search node budget per move (generation) [400]
  --eval-nodes N    node budget for promotion/progression matches [400]
  --steps N         optimizer steps per generation [2000]
  --pairs N         promotion match pairs [30]
  --opening-plies N shared random opening plies per pair [2]
  --epsilon F       exploration probability [0.1]
  --replay N        FIFO replay window in generations [4]
  --seed N          base seed; chunk c uses seed+c-1 [1]
  --threads N       worker threads [7]
  --status          print campaign progress and exit
  -h, --help        this help
EOF
}

[ $# -ge 1 ] || { usage; exit 1; }
NAME=$1; shift
DIR="campaigns/$NAME"

# Defaults (overridden by flags, then frozen into campaign.env).
GAME=fc-full WIDTH=64 CHUNKS=6 GENS=8 GAMES=200 NODES=400 EVAL_NODES=400
STEPS=2000 PAIRS=30 OPENING=2 EPSILON=0.1 REPLAY=4 SEED=1 THREADS=7
STATUS=0 FROM=""

while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM=$2; shift 2;;
    --game) GAME=$2; shift 2;;
    --width) WIDTH=$2; shift 2;;
    --chunks) CHUNKS=$2; shift 2;;
    --gens-per-chunk) GENS=$2; shift 2;;
    --games) GAMES=$2; shift 2;;
    --nodes) NODES=$2; shift 2;;
    --eval-nodes) EVAL_NODES=$2; shift 2;;
    --steps) STEPS=$2; shift 2;;
    --pairs) PAIRS=$2; shift 2;;
    --opening-plies) OPENING=$2; shift 2;;
    --epsilon) EPSILON=$2; shift 2;;
    --replay) REPLAY=$2; shift 2;;
    --seed) SEED=$2; shift 2;;
    --threads) THREADS=$2; shift 2;;
    --status) STATUS=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown option: $1" >&2; usage; exit 1;;
  esac
done

case "$GAME" in
  fc-tiny) RULESET=tiny;; fc-small) RULESET=small;;
  fc-medium) RULESET=medium;; fc-full) RULESET=full;;
  *) echo "unsupported --game $GAME" >&2; exit 1;;
esac

done_chunks() { find "$DIR" -mindepth 2 -maxdepth 2 -path '*/chunk_*/DONE' 2>/dev/null | wc -l; }

if [ "$STATUS" = 1 ]; then
  [ -d "$DIR" ] || { echo "no campaign at $DIR"; exit 1; }
  # shellcheck disable=SC1091
  source "$DIR/campaign.env"
  echo "campaign $NAME: $(done_chunks)/$CHUNKS chunks done ($GAME w$WIDTH, $GENS gens x $GAMES games/chunk)"
  grep '^chunk ' "$DIR/log.txt" 2>/dev/null || true
  exit 0
fi

if [ -f "$DIR/campaign.env" ]; then
  # Resume: frozen parameters win, except --chunks may extend the target.
  if [ -n "$FROM" ]; then
    echo "--from is only valid when creating a campaign; to continue $NAME" >&2
    echo "with its own recipe use --chunks, or fork with a NEW name." >&2
    exit 1
  fi
  TARGET=$CHUNKS
  # shellcheck disable=SC1091
  source "$DIR/campaign.env"
  if [ "$TARGET" -gt "$CHUNKS" ]; then CHUNKS=$TARGET; sed -i "s/^CHUNKS=.*/CHUNKS=$CHUNKS/" "$DIR/campaign.env"; fi
  echo "resuming campaign $NAME at chunk $(( $(done_chunks) + 1 ))/$CHUNKS"
else
  FROM_CHECKPOINT="" FROM_REPLAY=""
  if [ -n "$FROM" ]; then
    if [ -d "$FROM/champion" ]; then
      # A campaign: fork its champion and the latest chunk's replay.
      FROM_CHECKPOINT="$FROM/champion"
      LAST=$(find "$FROM" -mindepth 2 -maxdepth 2 -path '*/chunk_*/DONE' 2>/dev/null | sort | tail -1 | xargs -r dirname)
      if [ -n "$LAST" ] && [ -f "$LAST/replay.jsonl" ]; then FROM_REPLAY="$LAST/replay.jsonl"; fi
    elif [ -d "$FROM/checkpoint" ]; then
      # A chunk snapshot.
      FROM_CHECKPOINT="$FROM/checkpoint"
      if [ -f "$FROM/replay.jsonl" ]; then FROM_REPLAY="$FROM/replay.jsonl"; fi
    elif [ -f "$FROM/model.json" ]; then
      # A bare checkpoint (no replay to inherit).
      FROM_CHECKPOINT="$FROM"
    else
      echo "--from $FROM: not a campaign, chunk, or checkpoint directory" >&2
      exit 1
    fi
    SRC_WIDTH=$(python3 -c "import json;print(json.load(open('$FROM_CHECKPOINT/model.json'))['dims']['width'])")
    if [ "$SRC_WIDTH" != "$WIDTH" ]; then
      echo "--from model width is $SRC_WIDTH but --width is $WIDTH; pass --width $SRC_WIDTH" >&2
      exit 1
    fi
  fi
  mkdir -p "$DIR"
  cat > "$DIR/campaign.env" <<EOF
GAME=$GAME
RULESET=$RULESET
WIDTH=$WIDTH
CHUNKS=$CHUNKS
GENS=$GENS
GAMES=$GAMES
NODES=$NODES
EVAL_NODES=$EVAL_NODES
STEPS=$STEPS
PAIRS=$PAIRS
OPENING=$OPENING
EPSILON=$EPSILON
REPLAY=$REPLAY
SEED=$SEED
THREADS=$THREADS
FROM_CHECKPOINT=$FROM_CHECKPOINT
FROM_REPLAY=$FROM_REPLAY
EOF
  if [ -n "$FROM_CHECKPOINT" ]; then
    echo "campaign $NAME created: $GAME w$WIDTH, $CHUNKS chunks x $GENS gens x $GAMES games (base seed $SEED), forked from $FROM_CHECKPOINT"
  else
    echo "campaign $NAME created: $GAME w$WIDTH, $CHUNKS chunks x $GENS gens x $GAMES games (base seed $SEED)"
  fi
fi

cargo build --release --quiet 2>/dev/null || cargo build --release

while [ "$(done_chunks)" -lt "$CHUNKS" ]; do
  C=$(( $(done_chunks) + 1 ))
  CDIR=$(printf '%s/chunk_%03d' "$DIR" "$C")
  rm -rf "$CDIR"; mkdir -p "$CDIR"
  CHUNK_SEED=$(( SEED + C - 1 ))
  {
    echo "seed = $CHUNK_SEED"
    echo "model_width = $WIDTH"
    echo "generations = $GENS"
    echo "games_per_generation = $GAMES"
    echo "steps_per_generation = $STEPS"
    echo "epsilon = $EPSILON"
    echo "gen_node_budget = $NODES"
    echo "eval_node_budget = $EVAL_NODES"
    echo "replay_generations = $REPLAY"
    echo "promotion = \"match\""
    echo "promotion_pairs = $PAIRS"
    echo "opening_plies = $OPENING"
    echo "threads = $THREADS"
    if [ "$C" -gt 1 ]; then
      PREV=$(printf '%s/chunk_%03d' "$DIR" "$((C - 1))")
      echo "init_checkpoint = \"$PREV/checkpoint\""
      # Persisted FIFO replay window: without it the first generations
      # of a chunk train on one generation of data and regress.
      if [ -f "$PREV/replay.jsonl" ]; then
        echo "init_replay = \"$PREV/replay.jsonl\""
      fi
    elif [ -n "${FROM_CHECKPOINT:-}" ]; then
      # Forked campaign: chunk 1 continues from the source champion
      # (and its replay window, when the source had one).
      echo "init_checkpoint = \"$FROM_CHECKPOINT\""
      if [ -n "${FROM_REPLAY:-}" ]; then
        echo "init_replay = \"$FROM_REPLAY\""
      fi
    fi
    echo ""
    echo "[game]"
    echo "kind = \"forward_chess\""
    echo "ruleset = \"$RULESET\""
  } > "$CDIR/config.toml"

  echo "chunk $C/$CHUNKS: starting (seed $CHUNK_SEED, $(date -u +%H:%M:%S))" | tee -a "$DIR/log.txt"
  ./target/release/lab selfplay "$CDIR/config.toml" 2>&1 | tee -a "$DIR/log.txt" | tee "$CDIR/lab_output.txt"
  RUN=$(grep -m1 '^run directory: ' "$CDIR/lab_output.txt" | cut -d' ' -f3)
  [ -n "$RUN" ] && [ -d "$RUN" ] || { echo "chunk $C: could not locate run directory" >&2; exit 1; }
  echo "$RUN" > "$CDIR/run_dir.txt"
  cp -r "$RUN/checkpoint" "$CDIR/checkpoint"
  cp "$RUN/summary.json" "$RUN/metrics.jsonl" "$RUN/resolved.toml" "$CDIR/" 2>/dev/null || true
  cp "$RUN/replay.jsonl" "$CDIR/" 2>/dev/null || true
  if [ "$C" = 1 ]; then
    # Elo anchor: the campaign's untrained random-init net.
    mkdir -p "$DIR/baseline_gen0"
    cp "$RUN"/checkpoints/gen_000* "$DIR/baseline_gen0/" 2>/dev/null || true
    cp "$RUN/checkpoint/model.json" "$DIR/baseline_gen0/model.json"
    # gen_000 is saved as a bare file; normalize to checkpoint layout.
    if [ -f "$DIR/baseline_gen0/gen_000.bin" ]; then
      mv "$DIR/baseline_gen0/gen_000.bin" "$DIR/baseline_gen0/model.bin"
    fi
  fi
  rm -rf "$DIR/champion"; cp -r "$CDIR/checkpoint" "$DIR/champion"
  FINAL=$(python3 -c "import json;s=json.load(open('$CDIR/summary.json'));f=s.get('final_vs_gen0') or {};print(f\"score {f.get('score','?')} lcb {f.get('score_lcb95','?')} over {f.get('games','?')} games, mean plies {f.get('mean_plies','?')}\")" 2>/dev/null || echo "summary unavailable")
  echo "chunk $C/$CHUNKS: done — vs chunk-start champion: $FINAL" | tee -a "$DIR/log.txt"
  touch "$CDIR/DONE"
  if grep -q '^halting:' "$CDIR/lab_output.txt"; then
    echo "chunk $C halted (two consecutive regressions, plan §12.6)." | tee -a "$DIR/log.txt"
    echo "Campaign paused for diagnosis — inspect $CDIR/lab_output.txt and" | tee -a "$DIR/log.txt"
    echo "the rating curve; resume deliberately with: tools/fc_train.sh $NAME" | tee -a "$DIR/log.txt"
    exit 3
  fi
done

echo "campaign $NAME complete: $(done_chunks)/$CHUNKS chunks. Champion: $DIR/champion"
echo "Rate it: tools/fc_rating.py --campaign $DIR"
