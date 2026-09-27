#!/usr/bin/env bash
# Kick off a distillation run from everything captured so far (approach_docs/035).
#
#   scripts/distill.sh --model unsloth/gemma-3-12b-it-unsloth-bnb-4bit [train_lora args...]
#   DISTILL_BASE_MODEL=... scripts/distill.sh
#   DISTILL_BUILD_ARGS="--no-cloud" scripts/distill.sh --model ...
#
# 1. builds a fresh snapshot (scripts/build_distill_dataset.py -> research/finetune/datasets/)
# 2. refuses to train while the GPU is busy: training and LM Studio share one 16 GB card, so
#    unload the model in LM Studio first (GADS cannot run local workflows during training)
# 3. trains a LoRA on the snapshot with scripts/train_lora.py in .venv-train, logged to MLflow
#
# Afterwards: merge + convert to GGUF, serve it under its own GADS_ENGINE_TAG, and score it
# with scripts/eval_coder.py --tier 1,2 --repeats 3 against the base model (035 §6, §8).
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${DISTILL_BASE_MODEL:-}"
FORCE=0
PASS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) PASS+=("$1"); shift ;;
  esac
done
[[ -n "$MODEL" ]] || { echo "need --model <HF base checkpoint> (or DISTILL_BASE_MODEL)"; exit 2; }
[[ -x .venv-train/bin/python ]] || { echo "missing .venv-train (see scripts/validate_finetune_stack.py)"; exit 2; }

set -a; source .env; set +a
echo "== building snapshot"
# shellcheck disable=SC2086
PYTHONPATH=src uv run python scripts/build_distill_dataset.py ${DISTILL_BUILD_ARGS:-}
SNAP="research/finetune/datasets/$(readlink research/finetune/datasets/latest)"

if command -v nvidia-smi >/dev/null; then
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [[ "$USED" -gt 2000 && "$FORCE" -ne 1 ]]; then
    echo "GPU has ${USED} MiB in use (LM Studio still loaded?). Unload it, or pass --force."
    echo "Snapshot is built and kept: $SNAP"
    exit 3
  fi
fi

RUN="distill-$(basename "$SNAP")"
echo "== training $MODEL on $SNAP (run $RUN)"
.venv-train/bin/python scripts/train_lora.py --model "$MODEL" \
  --data "$SNAP/sft_train.jsonl" --eval "$SNAP/sft_holdout.jsonl" \
  --out "research/finetune/adapters/$RUN" --run-name "$RUN" "${PASS[@]}"
