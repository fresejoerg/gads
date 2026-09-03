"""Harvest contract-verified Coder traces into a fine-tuning corpus (approach_docs/031 §7 P1).

Read-only. Joins two stores that each hold half of a training example:

  Langfuse `observations` — the rendered prompt and completion, per ATTEMPT.
      Since 3f1ad49 (telemetry plan 010 P1+P2) `GADS_UNIFIED_COMPLETION` defaults true, so
      every model routes through `get_structured_completion`, where the user message IS the
      serialized `CoderInput` and `llm.py` stamps task_id / attempt / prompt_version /
      engine_id onto the generation. LiteLLM nests those under `requester_metadata`.

  GADS Postgres `task` — the VERDICT: status, the accepted `result_json['code']`, and
      `model_used` (which carries `native_fallback:` / `cloud_fallback:` / `native_primary:`
      prefixes). A task reaching status='completed' passed `validate_contract`, so the label
      is a real postcondition check, not a heuristic and not an LLM judge. That is what makes
      this rejection-sampling distillation rather than imitation of whatever was emitted.

Measured 2026-09-03: 380 completed-with-code tasks, 313 (82%) joinable, yielding 313
accepted completions and 338 rejected siblings across 169 multi-attempt tasks.

    PYTHONPATH=src uv run python scripts/harvest_coder_traces.py --out research/finetune/
    ... --mode dpo --local-only --engine qwen3.8-27b

Needs GADS_DATABASE_URL and GADS_LANGFUSE_DB_URL (both read-only here).

**Holdout is by SPEC, never by trace** — multiple runs of one spec share prompts and code
near-verbatim, so a trace-level split leaks catastrophically at this redundancy.
"""
import argparse
import collections
import json
import os
import pathlib
import random
import re
import sys

sys.path.insert(0, "src")

from sqlalchemy import create_engine, text

# Prefixes on `model_used` meaning the code was NOT written by the assigned model.
# Training on these teaches imitation of a hardcoded function (approach_docs/031 §4).
NON_MODEL_PREFIXES = ("native_fallback:", "native_primary:", "cloud_fallback:")

FENCE = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL)


def _unwrap(raw):
    """Recover the code from a Coder completion.

    The completion is either instructor-style JSON ({"code": ...}) or a fenced block,
    depending on the path that produced it. Returns None when neither yields a program —
    an empty or truncated generation is not a training target.
    """
    if not raw or not raw.strip():
        return None
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and obj.get("code"):
            return str(obj["code"])
    except Exception:
        pass
    m = FENCE.search(raw)
    if m:
        return m.group(1)
    return raw if "import " in raw or "def " in raw else None


def load_verdicts(engine):
    """task_id -> verdict, for tasks that PASSED their postcondition contract."""
    sql = """
        select t.id::text, t.description, t.result_json->>'code',
               coalesce(t.result_json->>'model_used','?'),
               p.id::text, p.last_state_json->>'spec_filename'
        from task t join project p on p.id = t.project_id
        where t.status='completed' and coalesce(t.result_json->>'code','') <> ''
    """
    out = {}
    with engine.connect() as c:
        for tid, desc, code, model, pid, spec in c.execute(text(sql)):
            out[tid] = {"task_id": tid, "description": desc, "accepted_code": code,
                        "model_used": model, "project_id": pid,
                        # Spec is the holdout unit; fall back to project so an untagged
                        # run still cannot straddle the split.
                        "spec": spec or f"project:{pid}"}
    return out


def load_generations(engine, verdict_ids):
    """task_id -> [attempts], each with the rendered messages and the emitted code."""
    sql = """
        select metadata->'requester_metadata'->>'task_id',
               (metadata->'requester_metadata'->>'attempt')::int,
               metadata->'requester_metadata'->>'prompt_version',
               metadata->'requester_metadata'->>'engine_id',
               input->'messages', output->>'content', prompt_tokens, completion_tokens
        from observations
        where name='CodeGenerator'
          and metadata->'requester_metadata'->>'task_id' is not null
        order by 2
    """
    by_task = collections.defaultdict(list)
    with engine.connect() as c:
        for tid, attempt, pv, eng, msgs, content, ptok, ctok in c.execute(text(sql)):
            if tid not in verdict_ids or not msgs:
                continue
            by_task[tid].append({
                "attempt": attempt or 1, "prompt_version": pv, "engine_id": eng,
                "messages": msgs, "code": _unwrap(content),
                "prompt_tokens": ptok or 0, "completion_tokens": ctok or 0,
            })
    return by_task


def build(verdicts, gens, args):
    """Pair verdicts with generations. Returns (examples, rejected_pairs, stats)."""
    sft, dpo, stats = [], [], collections.Counter()

    for tid, v in verdicts.items():
        attempts = sorted(gens.get(tid, []), key=lambda a: a["attempt"])
        if not attempts:
            stats["no_matching_generation"] += 1
            continue
        if v["model_used"].startswith(NON_MODEL_PREFIXES):
            stats["excluded_not_model_written"] += 1
            continue
        if args.local_only and v["model_used"] != "local_model":
            stats["excluded_not_local"] += 1
            continue

        # TARGET = the code the DB recorded, NOT the raw generation. Codegen output passes
        # through `executor._sanitize_code` before execution, so the two differ (measured:
        # 89 of 300 do) and only the sanitized form is what `validate_contract` actually
        # approved. Training on the raw generation would teach the model to reproduce text
        # that needed repairing before it could run.
        target = (v["accepted_code"] or "").strip()
        if not target:
            stats["no_parseable_completion"] += 1
            continue

        # PROMPT = which attempt's context to pair that target with.
        #   first — the clean prompt, no accumulated error feedback. Teaches "get it right
        #           immediately", which is the actual objective and the standard
        #           rejection-sampling formulation (009 §1: target the corrected version,
        #           not the model's own error-prone tendencies).
        #   final — the prompt that actually preceded the accepted code, error feedback
        #           included. Faithful to how the answer was reached; teaches repair.
        accepted = attempts[0] if args.pair_with == "first" else attempts[-1]
        stats[f"paired_with_{args.pair_with}_attempt"] += 1
        if len(attempts) > 1:
            stats["  ..of which needed retries"] += 1

        if args.engine and (accepted.get("engine_id") or "").split("@")[0] != args.engine:
            stats["excluded_wrong_engine"] += 1
            continue
        total_tok = accepted["prompt_tokens"] + accepted["completion_tokens"]
        if total_tok > args.max_tokens:
            # Truncating teaches the model to answer without context it WILL be given at
            # serve time. Dropping is the honest option (approach_docs/031 §6).
            stats["excluded_over_seq_len"] += 1
            continue

        rec = {"task_id": tid, "spec": v["spec"], "model_used": v["model_used"],
               "engine_id": accepted.get("engine_id"),
               "prompt_version": accepted.get("prompt_version"),
               "attempt": accepted["attempt"], "n_attempts": len(attempts),
               "messages": accepted["messages"], "completion": target,
               "prompt_tokens": accepted["prompt_tokens"]}
        sft.append(rec)

        # Preference pairs: every attempt the model made that did NOT become the accepted
        # code is a rejected sample against the validated target, under that attempt's own
        # context (chosen and rejected must share a prompt for DPO to be well-formed).
        for a in attempts[:-1]:
            if not a["code"] or a["code"].strip() == target:
                continue
            dpo.append({"task_id": tid, "spec": v["spec"],
                        "engine_id": a.get("engine_id"),
                        "messages": a["messages"],
                        "rejected": a["code"], "chosen": target})
    return sft, dpo, stats


def cap_per_spec(records, cap, seed):
    """Cap examples per spec so the corpus is not dominated by the most-run workflow."""
    if not cap:
        return records
    rng = random.Random(seed)
    grouped = collections.defaultdict(list)
    for r in records:
        grouped[r["spec"]].append(r)
    out = []
    for spec, rs in grouped.items():
        rng.shuffle(rs)
        out.extend(rs[:cap])
    return out


def split_by_spec(records, holdout, seed):
    specs = sorted({r["spec"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(specs)
    n_hold = max(1, int(len(specs) * holdout)) if len(specs) > 1 else 0
    held = set(specs[:n_hold])
    return ([r for r in records if r["spec"] not in held],
            [r for r in records if r["spec"] in held], held)


def write_jsonl(path, rows, key):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(key(r)) + "\n")
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="research/finetune", help="output directory")
    ap.add_argument("--mode", choices=["sft", "dpo", "both"], default="both")
    ap.add_argument("--local-only", action="store_true",
                    help="keep only local_model completions (excludes cloud)")
    ap.add_argument("--engine", default=None,
                    help="keep only traces from this served engine, e.g. gemma-4-12b. "
                         "Traces predating the engine_id stamp have none and are dropped.")
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="drop examples whose prompt+completion exceeds this (default 8192; "
                         "p95 of the measured corpus is 10,653)")
    ap.add_argument("--pair-with", choices=["first", "final"], default="first",
                    help="which attempt's prompt to pair the validated code with "
                         "(default first: teach getting it right without error feedback)")
    ap.add_argument("--cap-per-spec", type=int, default=40)
    ap.add_argument("--holdout", type=float, default=0.2, help="fraction of SPECS held out")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    gads_url = os.environ.get("GADS_DATABASE_URL")
    lf_url = os.environ.get("GADS_LANGFUSE_DB_URL")
    if not gads_url or not lf_url:
        sys.exit("Need GADS_DATABASE_URL and GADS_LANGFUSE_DB_URL (see .env).")

    verdicts = load_verdicts(create_engine(gads_url, pool_pre_ping=True))
    print(f"GADS  : {len(verdicts)} completed tasks with code (contract-validated)")
    gens = load_generations(create_engine(lf_url, pool_pre_ping=True), set(verdicts))
    print(f"Langfuse: {sum(len(v) for v in gens.values())} generations "
          f"across {len(gens)} of those tasks ({100*len(gens)//max(len(verdicts),1)}% joined)")

    sft, dpo, stats = build(verdicts, gens, args)
    print("\nfiltering:")
    for k, n in stats.most_common():
        print(f"  {n:5d}  {k}")

    sft = cap_per_spec(sft, args.cap_per_spec, args.seed)
    train, held, held_specs = split_by_spec(sft, args.holdout, args.seed)
    out = pathlib.Path(args.out)

    print(f"\nSFT   : {len(sft)} examples over {len({r['spec'] for r in sft})} specs")
    if args.mode in ("sft", "both"):
        n1 = write_jsonl(out / "sft_train.jsonl", train,
                         lambda r: {"messages": r["messages"] + [
                             {"role": "assistant", "content": r["completion"]}]})
        n2 = write_jsonl(out / "sft_holdout.jsonl", held,
                         lambda r: {"messages": r["messages"] + [
                             {"role": "assistant", "content": r["completion"]}]})
        print(f"  -> {out/'sft_train.jsonl'} ({n1})  |  holdout ({n2}) "
              f"over {len(held_specs)} held-out spec(s)")
    if args.mode in ("dpo", "both"):
        dpo = [d for d in dpo if d["spec"] not in held_specs]
        n3 = write_jsonl(out / "dpo_train.jsonl", dpo,
                         lambda r: {"messages": r["messages"],
                                    "chosen": r["chosen"], "rejected": r["rejected"]})
        print(f"  -> {out/'dpo_train.jsonl'} ({n3} preference pairs)")
        if not args.engine:
            print("  ! DPO pairs are only on-policy if the rejected side came from the model "
                  "you are training. Pre-engine_id traces are an unattributable mixture "
                  "(approach_docs/031 §5) — pass --engine once stamped traces exist.")

    write_jsonl(out / "manifest.jsonl", sft,
                lambda r: {k: r[k] for k in ("task_id", "spec", "model_used", "engine_id",
                                             "prompt_version", "attempt", "n_attempts",
                                             "prompt_tokens")})
    print(f"  -> {out/'manifest.jsonl'} (provenance for every example)")


if __name__ == "__main__":
    main()
