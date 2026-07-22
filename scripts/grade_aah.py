#!/usr/bin/env python3
"""Grade a GADS run of the aah grounding benchmark.

Reads the run's answer artifact (aah_answers.json = {question_id: answer_text}) and
scores it against research/benchmarks/aah_v1/expected.json. Three graders, picked per
question (ported verbatim from d-n-ust/ai-analytics-harness, MIT):
  - numeric:    extract a number; correct within tolerance (a rate as % is accepted)
  - keywords:   the answer names the right metric
  - diagnostic: the answer identifies the right driver AND the root cause

Reports accuracy overall and per tier (lookup/filtered/metric/knowledge/diagnostic),
plus abstention and confident-wrong counts — so a GADS grounding run is directly
comparable to aah's 41%->92% curve.

Usage:
  python scripts/grade_aah.py <aah_answers.json> [--rung N] [--json]
"""
from __future__ import annotations
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path

EXPECTED = Path(__file__).resolve().parent.parent / "research" / "benchmarks" / "aah_v1" / "expected.json"
TIER_ORDER = ["lookup", "filtered", "metric", "knowledge", "diagnostic"]


def extract_number(text):
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return float(text)
    m = re.search(r"-?\d+\.?\d*", str(text).replace(",", "").replace("$", ""))
    return float(m.group()) if m else None


def _close(x, gold, tol):
    return abs(x - gold) / max(abs(gold), 1e-9) <= tol


def grade_numeric(answer_text, gold, tol):
    x = extract_number(answer_text)
    if x is None or gold is None:
        return {"executed": x is not None, "correct": False, "extracted": x}
    candidates = [gold, gold * 100] if abs(gold) < 1 else [gold]  # rate answered as %
    return {"executed": True, "correct": any(_close(x, g, tol) for g in candidates), "extracted": x}


def grade_keywords(text, keywords):
    t = (text or "").lower()
    return {"executed": True, "correct": any(k.lower() in t for k in keywords)}


def grade_diagnostic(text, spec):
    t = (text or "").lower()
    driver_ok = any(k.lower() in t for k in spec.get("driver", []))
    cause_ok = any(k.lower() in t for k in spec.get("cause", []))
    return {"executed": True, "correct": driver_ok and cause_ok, "driver_ok": driver_ok, "cause_ok": cause_ok}


ABSTAIN_RE = re.compile(r"\b(cannot|can't|can not|unable|not able|no answer|insufficient)\b", re.I)


def grade_one(answer_text, q):
    grader = q.get("grader", "numeric")
    if grader == "diagnostic":
        g = grade_diagnostic(answer_text, q["gold_diagnostic"])
    elif grader == "keywords":
        g = grade_keywords(answer_text, q["gold_keywords"])
    else:
        g = grade_numeric(answer_text, q.get("gold"), q.get("tolerance", 0.02))
    abstained = bool(answer_text) and bool(ABSTAIN_RE.search(str(answer_text))) and extract_number(answer_text) is None
    g["abstained"] = abstained
    g["confident_wrong"] = (not abstained and not g["correct"] and answer_text not in (None, ""))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers", help="path to the run's aah_answers.json ({qid: answer_text})")
    ap.add_argument("--rung", type=int, default=None, help="grounding rung this run used (for the report)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    expected = json.loads(EXPECTED.read_text())["questions"]
    answers = json.loads(Path(args.answers).read_text())

    per_tier = defaultdict(lambda: {"n": 0, "correct": 0})
    n = correct = abstained = confident_wrong = missing = 0
    rows = []
    for qid, q in expected.items():
        ans = answers.get(qid)
        if ans is None:
            missing += 1
        g = grade_one(ans, q)
        n += 1
        correct += g["correct"]
        abstained += g["abstained"]
        confident_wrong += g["confident_wrong"]
        per_tier[q["tier"]]["n"] += 1
        per_tier[q["tier"]]["correct"] += g["correct"]
        rows.append((qid, q["tier"], g["correct"], g.get("extracted"), ans))

    report = {
        "answers_file": str(args.answers), "rung": args.rung,
        "n": n, "accuracy": round(correct / n, 4) if n else 0.0,
        "correct": correct, "missing": missing,
        "abstained": abstained, "confident_wrong": confident_wrong,
        "per_tier": {t: {"n": per_tier[t]["n"], "correct": per_tier[t]["correct"],
                         "accuracy": round(per_tier[t]["correct"] / per_tier[t]["n"], 4) if per_tier[t]["n"] else 0.0}
                     for t in TIER_ORDER if t in per_tier},
    }
    if args.json:
        print(json.dumps(report, indent=2))
        return

    rung = f" (rung {args.rung})" if args.rung is not None else ""
    print(f"aah grounding benchmark{rung} — {args.answers}")
    print(f"  overall accuracy: {report['accuracy']:.0%}  ({correct}/{n})")
    print(f"  abstained: {abstained}   confident-wrong: {confident_wrong}   missing answers: {missing}")
    print("  by tier:")
    for t in TIER_ORDER:
        if t in report["per_tier"]:
            pt = report["per_tier"][t]
            print(f"    {t:11s} {pt['correct']}/{pt['n']}  ({pt['accuracy']:.0%})")
    wrong = [(qid, ex, ans) for qid, tier, ok, ex, ans in rows if not ok]
    if wrong:
        print("  misses:")
        for qid, ex, ans in wrong:
            print(f"    {qid:30s} extracted={ex}  answer={str(ans)[:60]!r}")


if __name__ == "__main__":
    main()
