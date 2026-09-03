"""Stage-level Coder evaluation — tiers 0 and 1 (approach_docs/031 §7; 009 Phase 2).

Measures the Coder ALONE, on frozen held-out prompts. End-to-end benchmarks
(`score_benchmark.py`) cannot do this job: minutes per run, nondeterministic, and a Coder
gain is masked by a Planner wobble, so credit assignment is murky at the n we have.

  Tier 0  held-out loss / perplexity. Sanity only — a model can lower loss while writing
          worse code, so this answers "is it learning", never "is it better".
  Tier 1  static checks on GENERATED code, no execution:
            parses          — ast.parse succeeds. Catches the truncation and repetition
                              pathologies 009 §1 classed as UNREACHABLE by SFT, so a flat
                              line here is itself the finding.
            no_banned_import— hallucinated `gads_utils` / `gads_helpers` / `causal_models`
            no_mock_data    — the executor's own hallucination tokens (server.py ~2502)
            no_sentinel     — code printing a raw GADS_*_JSON: prefix corrupts the parser
            binds_required  — does it bind the `required_variables` its contract demands?
                              (carried in the holdout file; only recipe-compiled nodes
                              declare them, so this scores a subset)

Tier 2 (execute the code and judge with the real `validate_contract`) is the instrument
that can support a capability claim; this is not it. With ~35 held-out examples a 10-point
move sits inside the noise, so treat these as regression tripwires, not evidence of gain.

    # validate the checks themselves against the reference solutions (no model needed)
    PYTHONPATH=src uv run python scripts/eval_coder.py --reference-only

    # measure what LM Studio currently serves
    PYTHONPATH=src uv run python scripts/eval_coder.py --backend litellm --tag base

    # measure a local checkpoint/adapter before GGUF conversion
    .venv-train/bin/python scripts/eval_coder.py --backend hf \\
        --model unsloth/... --adapter out/lora --tier 0,1 --tag tuned
"""
import argparse
import ast
import json
import os
import pathlib
import re
import sys
import time

sys.path.insert(0, "src")

BANNED_IMPORTS = ("gads_utils", "gads_helpers", "causal_models", "causal_inference_lib")
# Kept in sync with server.py's HALLUCINATION GUARD by intent, not by import: this scores
# the CODE before it runs, that guard scores stdout after.
MOCK_TOKENS = ("simulating data", "mock data", "dummy data", "no data available",
               "environment is empty", "no files provided")
SENTINELS = ("GADS_INSIGHTS_JSON:", "GADS_FLOOR_JSON:", "GADS_STATE_SNAPSHOT:",
             "GADS_HYPOTHESIS_JSON:", "GADS_METRICS_JSON:")
FENCE = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL)


def _load_sanitizer():
    """The executor's own repair pass, so eval matches what production actually runs.

    The training targets are POST-sanitizer (the DB stores the code that executed), so
    scoring raw generations against them compares against a repaired ceiling and
    understates the served model. Reporting both is the honest form: the gap between
    them is precisely how much the harness is currently rescuing.
    """
    try:
        from gads.core.executor import _sanitize_code
        return _sanitize_code
    except Exception:        # training venv lacks the runtime deps
        return None


SANITIZE = _load_sanitizer()


def unwrap(raw):
    if not raw or not raw.strip():
        return ""
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and obj.get("code"):
            return str(obj["code"])
    except Exception:
        pass
    m = FENCE.search(raw)
    return m.group(1) if m else raw


def bound_names(tree):
    """Every name the module binds at any depth: assignment, tuple unpack, for, with, def."""
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.alias):
            out.add((n.asname or n.name).split(".")[0])
    return out


def score(code, required):
    """Tier-1 checks for one completion. `binds_required` is None when nothing is demanded,
    so a node with no declared outputs cannot inflate the pass rate."""
    r = {"chars": len(code)}
    try:
        ast.parse(code)
        r["parses_raw"] = True
    except SyntaxError:
        r["parses_raw"] = False
    if SANITIZE:
        code = SANITIZE(code)
    try:
        tree = ast.parse(code)
        r["parses"] = True
    except SyntaxError as e:
        # Every downstream check needs an AST or a running program; scoring them on
        # unparseable text would invent signal. None = not applicable, not failed.
        # The reason matters: a truncation is a decoding pathology SFT cannot reach
        # (009 §1), while prose leakage is a formatting habit it can.
        return {**r, "parses": False, "syntax_error": f"{e.msg} (line {e.lineno})",
                "empty": not code.strip(),
                "no_banned_import": None, "no_mock_data": None,
                "no_sentinel": None, "binds_required": None, "missing": []}
    low = code.lower()
    r["no_banned_import"] = not any(b in low for b in BANNED_IMPORTS)
    r["no_mock_data"] = not any(t in low for t in MOCK_TOKENS)
    r["no_sentinel"] = not any(f'"{s}' in code or f"'{s}" in code for s in SENTINELS)
    if required:
        names = bound_names(tree)
        missing = sorted(set(required) - names)
        r["binds_required"], r["missing"] = not missing, missing
    else:
        r["binds_required"], r["missing"] = None, []
    return r


def load_contracts(task_ids):
    try:
        from sqlalchemy import create_engine, text
    except ImportError:      # training venv has no DB driver; the holdout is self-describing
        return {}
    url = os.environ.get("GADS_DATABASE_URL")
    if not url:
        print("  ! GADS_DATABASE_URL unset — skipping binds_required", file=sys.stderr)
        return {}
    out = {}
    with create_engine(url, pool_pre_ping=True).connect() as c:
        for tid, pc in c.execute(text(
                "select id::text, postcondition_json from task where status='completed'")):
            if tid in task_ids and pc:
                out[tid] = list((pc or {}).get("required_variables") or [])
    return out


def gen_litellm(rows, args):
    import httpx
    base = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000/v1")
    key = os.environ.get("LITELLM_MASTER_KEY", "sk-1234")
    outs = []
    for i, r in enumerate(rows, 1):
        msgs = [m for m in r["messages"] if m["role"] != "assistant"]
        try:
            resp = httpx.post(f"{base.rstrip('/')}/chat/completions",
                              headers={"Authorization": f"Bearer {key}"},
                              json={"model": args.model or "local_model", "messages": msgs,
                                    "temperature": args.temperature,
                                    "max_tokens": args.max_new_tokens},
                              timeout=args.timeout)
            resp.raise_for_status()
            outs.append(resp.json()["choices"][0]["message"]["content"] or "")
        except Exception as e:
            print(f"  [{i}/{len(rows)}] generation failed: {type(e).__name__}: "
                  f"{str(e)[:100]}", flush=True)
            outs.append("")
        if i % 5 == 0:
            print(f"  generated {i}/{len(rows)}", flush=True)
    return outs


def load_hf(args):
    """Load base (+ optional adapter) for tier 0 and/or local generation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    kw = {}
    if args.load_in_4bit:
        # transformers 5.x dropped the bare `load_in_4bit=` kwarg. An already-quantized
        # checkpoint carries its own config and needs neither.
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", **kw)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    return model, tok


def tier0_loss(rows, model, tok, args):
    """Mean token-level NLL on the assistant span only.

    Prompt tokens are masked: scoring them would measure how well the model reproduces
    schemas and kernel snapshots, which is not the thing being trained.
    """
    import torch
    tot_nll, tot_tok = 0.0, 0
    for i, r in enumerate(rows, 1):
        msgs = r["messages"]
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False,
                                         add_generation_prompt=True)
        full = prompt + msgs[-1]["content"]
        p_ids = tok(prompt, return_tensors="pt").input_ids
        f_ids = tok(full, return_tensors="pt", truncation=True,
                    max_length=args.max_seq_len).input_ids.to(model.device)
        n_prompt = min(p_ids.shape[1], f_ids.shape[1] - 1)
        labels = f_ids.clone()
        labels[:, :n_prompt] = -100
        with torch.no_grad():
            out = model(f_ids, labels=labels)
        n = int((labels != -100).sum())
        if n:
            tot_nll += float(out.loss) * n
            tot_tok += n
        if i % 10 == 0:
            print(f"  scored {i}/{len(rows)}", flush=True)
    import math
    mean = tot_nll / max(tot_tok, 1)
    return {"loss": round(mean, 4), "perplexity": round(math.exp(min(mean, 20)), 2),
            "completion_tokens_scored": tot_tok}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout", default="research/finetune/sft_holdout.jsonl")
    ap.add_argument("--tier", default="1", help="comma list: 0, 1 (default 1)")
    ap.add_argument("--backend", choices=["litellm", "hf"], default="litellm")
    ap.add_argument("--reference-only", action="store_true",
                    help="score the REFERENCE solutions instead of generating. Establishes "
                         "the ceiling and validates the checks — a check the references "
                         "fail is a broken check, not a model finding.")
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--tag", default=None, help="label for this run in the report")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=1,
                    help="re-run generation N times to establish the NOISE FLOOR. Local "
                         "serving is not deterministic even at temperature 0 (measured: "
                         "74.3%% vs 65.7%% parse rate on identical inputs), so a single "
                         "run cannot support a pre/post comparison.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=8192,
                    help="match llm.py's local-model budget (8192); a smaller cap would "
                         "manufacture truncation and score it as a model defect")
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", default="research/finetune/eval_report.jsonl")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.holdout)]
    if args.limit:
        rows = rows[:args.limit]
    tiers = {t.strip() for t in args.tier.split(",")}
    tag = args.tag or ("reference" if args.reference_only else (args.model or "local_model"))
    print(f"holdout: {len(rows)} examples over "
          f"{len({r.get('spec') for r in rows})} spec(s) | tag={tag}")

    # The holdout carries its own contracts (harvest writes them); the DB join is only a
    # fallback for files produced before that.
    contracts = {r["task_id"]: r["required_variables"] for r in rows
                 if r.get("required_variables")}
    if not contracts:
        contracts = load_contracts({r.get("task_id") for r in rows})
    n_scored = sum(1 for r in rows if contracts.get(r.get("task_id")))
    print(f"holdout examples declaring contract variables: {n_scored}/{len(rows)}")

    report = {"tag": tag, "n": len(rows), "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    model = tok = None
    if "0" in tiers or (args.backend == "hf" and not args.reference_only):
        if not args.model:
            sys.exit("--backend hf / --tier 0 needs --model")
        print("\nloading model ...", flush=True)
        model, tok = load_hf(args)

    if "0" in tiers:
        print("\nTIER 0 — held-out loss (assistant span only)")
        report["tier0"] = tier0_loss(rows, model, tok, args)
        print(f"  loss {report['tier0']['loss']} | ppl {report['tier0']['perplexity']}")

    if "1" in tiers:
        print("\nTIER 1 — static checks on generated code"
              + (f" ({args.repeats} repeats for the noise floor)" if args.repeats > 1 else ""))
        gen_path = pathlib.Path(args.out).parent / f"generations_{tag.replace('/', '_')}.jsonl"
        runs = []

        for rep in range(args.repeats):
            if args.reference_only:
                codes = [r["messages"][-1]["content"] for r in rows]
            elif args.backend == "litellm":
                if args.repeats > 1:
                    print(f"  -- run {rep + 1}/{args.repeats}", flush=True)
                codes = [unwrap(c) for c in gen_litellm(rows, args)]
            else:
                sys.exit("hf generation not implemented — use --backend litellm, or "
                         "--reference-only. (Serve the merged checkpoint to compare "
                         "like-for-like.)")

            # Keep the LAST run's generations: they are the evidence for the failure
            # breakdown printed below.
            with open(gen_path, "w") as gf:
                for r, code in zip(rows, codes):
                    gf.write(json.dumps({"task_id": r.get("task_id"), "spec": r.get("spec"),
                                         "generation": code}) + "\n")

            results = []
            for r, code in zip(rows, codes):
                sc = score(code, contracts.get(r.get("task_id")))
                sc.update(task_id=r.get("task_id"), spec=r.get("spec"))
                results.append(sc)

            agg = {}
            for k in ("parses_raw", "parses", "no_banned_import", "no_mock_data",
                      "no_sentinel", "binds_required"):
                vals = [x[k] for x in results if x.get(k) is not None]
                agg[k] = {"pass": sum(vals), "n": len(vals),
                          "rate": round(sum(vals) / len(vals), 3) if vals else None}
            runs.append(agg)
            if args.reference_only:
                break      # deterministic; repeating adds nothing

        report["tier1_runs"] = runs
        report["tier1"] = runs[-1]
        keys = ("parses_raw", "parses", "no_banned_import", "no_mock_data",
                "no_sentinel", "binds_required")
        if len(runs) == 1:
            print(f"  {'check':18s} {'pass':>8s}  rate")
            for k in keys:
                v = runs[0][k]
                rate = "—" if v["rate"] is None else f"{v['rate']:.1%}"
                print(f"  {k:18s} {v['pass']:4d}/{v['n']:<4d}  {rate}")
        else:
            print(f"\n  {'check':18s} {'mean':>7s} {'min':>7s} {'max':>7s} {'spread':>8s}")
            for k in keys:
                rs = [r[k]["rate"] for r in runs if r[k]["rate"] is not None]
                if not rs:
                    print(f"  {k:18s} {'—':>7s}")
                    continue
                lo, hi = min(rs), max(rs)
                print(f"  {k:18s} {sum(rs)/len(rs):>6.1%} {lo:>7.1%} {hi:>7.1%} "
                      f"{hi - lo:>7.1%}")
            print("\n  The spread is the NOISE FLOOR: any pre/post difference smaller than "
                  "it\n  is not evidence. Local serving is nondeterministic even at "
                  "temperature 0.")

        rescued = sum(1 for x in results if x.get("parses") and not x.get("parses_raw"))
        if rescued:
            print(f"\n  {rescued} generation(s) parse ONLY after the executor's sanitizer "
                  f"— that gap is harness rescue, not model capability.")
        if SANITIZE is None:
            print("  ! sanitizer unavailable in this venv; `parses` == raw parse rate")
        broken = [x for x in results if x.get("parses") is False]
        if broken:
            print(f"\n  parse failures in the last run ({len(broken)}):")
            import collections as _c
            for msg, n in _c.Counter(
                    ("EMPTY generation" if x.get("empty") else x.get("syntax_error", "?"))
                    for x in broken).most_common(8):
                print(f"    {n:3d}x  {msg}")
            print(f"  generations saved -> {gen_path}")
        miss = [x for x in results if x.get("binds_required") is False]
        if miss:
            print(f"\n  contract variables never bound (top 5):")
            for x in miss[:5]:
                print(f"    {x['spec'][:38]:38s} missing {x['missing']}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        f.write(json.dumps(report) + "\n")
    print(f"\nappended -> {out}")


if __name__ == "__main__":
    main()
