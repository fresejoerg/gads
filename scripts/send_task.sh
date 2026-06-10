#!/usr/bin/env bash
# Usage: ./scripts/send_task.sh "your task description"
#        ./scripts/send_task.sh --to Deepfrese "your task description"
#
# Appends one JSON line to agent_inbox.jsonl.
# Both agents monitor this file and will act on messages addressed to them or "all".

INBOX="$(dirname "$0")/../agent_inbox.jsonl"
TO="all"

if [[ "$1" == "--to" ]]; then
  TO="$2"
  shift 2
fi

CONTENT="$*"
if [[ -z "$CONTENT" ]]; then
  echo "Usage: $0 [--to Deepfrese|Antigravity|all] \"task description\""
  exit 1
fi

TS=$(date -Iseconds)
LINE=$(printf '{"from":"GOD","to":"%s","ts":"%s","type":"task","content":"%s"}' \
  "$TO" "$TS" "${CONTENT//\"/\\\"}")

echo "$LINE" >> "$INBOX"
echo "Sent to $TO: $CONTENT"
