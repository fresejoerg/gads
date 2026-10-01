"""Coverage scoreboard: how far each task type in taxonomy.yaml is *demonstrated* — no LLM calls.

`test_vocabulary.py` answers "can the Router reach a recipe for this term?". This answers
the stronger question the coverage sprint is measured on: for each task family/subtype, is
the capability merely declared, or actually demonstrated?

    L0  nothing
    L1  a routable recipe declares this term SPECIFICALLY (family rows: the family or one of
        its subtypes). Reaching it only through a bare family label is "inherited" and does
        not count — that is routability, not a recipe built for the job.
    L2  L1 + a tagged spec exercises it
    L3  L2 + a benchmark for one of those specs with an EXTERNALLY anchored metric
        (`provenance: external | analytic`, approach_docs/033 §3a) AND a scored cloud PASS
        (research/benchmarks/results.jsonl, written by scripts/score_benchmark.py)
    L4  L3 + a scored local PASS

`L3*` / `L4*` are provisional: the benchmark scored PASS, but none of its metrics declares a
`provenance`, so whether it is anchored externally or on a past GADS run is unrecorded (033
P0 adds the field). A run that merely *completed* (dial ledger outcome=pass) is reported as
evidence but never raises the level — completion is not correctness.

    PYTHONPATH=src uv run python scripts/coverage_report.py              # table
    PYTHONPATH=src uv run python scripts/coverage_report.py --families   # family rows only
    PYTHONPATH=src uv run python scripts/coverage_report.py --json       # machine-readable
    PYTHONPATH=src uv run python scripts/coverage_report.py --save research/coverage/<date>.json
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

sys.path.insert(0, "src")

from gads.core import taxonomy as tx
from gads.core.knowledge import KnowledgeRegistry

RECIPES_DIR = "src/gads/knowledge/recipes"
BENCH_DIR = "research/benchmarks"
DIAL_LEDGER = "research/dial_ledger.jsonl"
ANCHORED = {"external", "analytic"}
LEVELS = ["L0", "L1", "L2", "L3*", "L3", "L4*", "L4"]


def _canon(term):
    return tx.canonical_task(term) or term


def _jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _spec_file(name: str) -> str:
    return name if name.endswith(".md") else name + ".md"


def load_benchmarks():
    """spec_file -> list of {id, anchor} where anchor is 'anchored' | 'gads_reference' | 'unclassified'."""
    by_spec = defaultdict(list)
    id_to_spec = {}
    for path in sorted(glob.glob(os.path.join(BENCH_DIR, "*", "expected.json"))):
        with open(path) as f:
            exp = json.load(f)
        bid = exp.get("benchmark_id") or os.path.basename(os.path.dirname(path))
        spec = exp.get("spec_file")
        provs = {m.get("provenance") for m in (exp.get("metrics") or {}).values() if isinstance(m, dict)}
        provs.discard(None)
        anchor = ("anchored" if provs & ANCHORED else
                  "gads_reference" if provs else "unclassified")
        if spec:
            by_spec[spec].append({"id": bid, "anchor": anchor})
            id_to_spec[bid] = spec
    return by_spec, id_to_spec


def load_scored(id_to_spec):
    """spec_file -> {'cloud': n_pass, 'local': n_pass} from scored benchmark results."""
    scored = defaultdict(Counter)
    for r in _jsonl(os.path.join(BENCH_DIR, "results.jsonl")):
        spec = id_to_spec.get(r.get("benchmark_id"))
        if spec and r.get("verdict") == "PASS":
            scored[spec]["local" if r.get("mode") == "local" else "cloud"] += 1
    return scored


def load_completions():
    """spec_file -> {'cloud': n, 'local': n} workflow completions (dial ledger outcome=pass)."""
    done = defaultdict(Counter)
    for r in _jsonl(DIAL_LEDGER):
        if r.get("spec") and r.get("outcome") == "pass":
            done[r["spec"]]["local" if r.get("routing_mode") == "local" else "cloud"] += 1
    return done


def build():
    vocab = tx.load_vocab()
    registry = KnowledgeRegistry(RECIPES_DIR)
    routable = {rid: r for rid, r in registry.recipes.items() if not registry.is_pin_only(r.applies_when)}
    pinned = {rid: r for rid, r in registry.recipes.items() if rid not in routable}
    declared = {rid: [_canon(t) for t in (r.applies_when or {}).get("task_type") or []]
                for rid, r in registry.recipes.items()}

    specs = [s for s in tx.spec_index() if s["tagged"]]
    bench_by_spec, id_to_spec = load_benchmarks()
    scored = load_scored(id_to_spec)
    completed = load_completions()

    def specific(rid, term):
        if "." in term:
            return term in declared[rid]
        return any(c == term or c.startswith(term + ".") for c in declared[rid])

    def reaches(rid, term):
        return any(tx.tasks_overlap(c, term) for c in declared[rid])

    rows = []
    for fam, subs in vocab["task"].items():
        for term in [fam] + [f"{fam}.{s}" for s in (subs or [])]:
            spec_rec = sorted(rid for rid in routable if specific(rid, term))
            inh_rec = sorted(rid for rid in routable if reaches(rid, term) and rid not in spec_rec)
            pin_rec = sorted(rid for rid in pinned if specific(rid, term) or reaches(rid, term))
            term_specs = sorted({_spec_file(s["spec"]) for s in specs
                                 if any(tx.tasks_overlap(t, term) for t in s["task"])})
            benches = [b for sp in term_specs for b in bench_by_spec.get(sp, [])]
            # A scored PASS only counts toward the level for specs that have a benchmark.
            sc = Counter()
            sc_anchored = Counter()
            for sp in term_specs:
                anchors = {b["anchor"] for b in bench_by_spec.get(sp, [])}
                sc.update(scored.get(sp, Counter()))
                if "anchored" in anchors:
                    sc_anchored.update(scored.get(sp, Counter()))
            comp = Counter()
            for sp in term_specs:
                comp.update(completed.get(sp, Counter()))

            level = "L0"
            if spec_rec:
                level = "L1"
                if term_specs:
                    level = "L2"
                    if sc_anchored["cloud"]:
                        level = "L4" if sc_anchored["local"] else "L3"
                    elif benches and sc["cloud"]:
                        level = "L4*" if sc["local"] else "L3*"
            rows.append({
                "term": term, "family": fam, "is_family": term == fam, "level": level,
                "specific_recipes": spec_rec, "inherited_recipes": inh_rec, "pin_only_recipes": pin_rec,
                "specs": term_specs,
                "benchmarks": sorted({b["id"] for b in benches}),
                "benchmark_anchors": dict(Counter(b["anchor"] for b in benches)),
                "scored_pass": {"cloud": sc["cloud"], "local": sc["local"]},
                "completed_runs": {"cloud": comp["cloud"], "local": comp["local"]},
            })

    modalities = {}
    for mod in vocab["modality"]:
        modalities[mod] = {
            "routable_recipes": sorted(rid for rid, r in routable.items()
                                       if any(tx.canonical_modality(m) == mod
                                              for m in (r.applies_when or {}).get("data_modality") or [])),
            "specs": sorted(_spec_file(s["spec"]) for s in specs if mod in (s.get("modality") or [])),
        }

    fam_rows = [r for r in rows if r["is_family"]]
    leaf_rows = [r for r in rows if not r["is_family"] or not vocab["task"].get(r["family"])]
    summary = {
        "families_by_level": dict(Counter(r["level"] for r in fam_rows)),
        "leaf_terms_by_level": dict(Counter(r["level"] for r in leaf_rows)),
        "families_total": len(fam_rows),
        "leaf_terms_total": len(leaf_rows),
        # Same definition as test_vocabulary.py check 6, for cross-checking.
        "leaf_terms_agent_reachable": sum(1 for r in leaf_rows if r["specific_recipes"] or r["inherited_recipes"]),
        "routable_recipes": len(routable),
        "pin_only_recipes": len(pinned),
        "tagged_specs": len(specs),
        "benchmarks": sum(len(v) for v in bench_by_spec.values()),
        "benchmark_anchors": dict(Counter(b["anchor"] for v in bench_by_spec.values() for b in v)),
        "modalities_with_routable_recipe": sum(1 for m in modalities.values() if m["routable_recipes"]),
        "modalities_total": len(modalities),
    }
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = None
    return {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "commit": commit, "summary": summary, "rows": rows, "modalities": modalities}


def _short(ids, n=2):
    if not ids:
        return ""
    return ", ".join(ids[:n]) + (f" +{len(ids) - n}" if len(ids) > n else "")


def print_table(report, families_only=False):
    print(f"{'task':40} {'lvl':4} {'spec':>4} {'inh':>3} {'specs':>5} {'bench':>5} "
          f"{'scored c/l':>10} {'done c/l':>8}  specific recipes")
    for r in report["rows"]:
        if families_only and not r["is_family"]:
            continue
        name = r["term"] if r["is_family"] else "  " + r["term"].split(".", 1)[1]
        sp, dn = r["scored_pass"], r["completed_runs"]
        print(f"{name:40} {r['level']:4} {len(r['specific_recipes']):>4} {len(r['inherited_recipes']):>3} "
              f"{len(r['specs']):>5} {len(r['benchmarks']):>5} {sp['cloud']:>5}/{sp['local']:<4} "
              f"{dn['cloud']:>3}/{dn['local']:<4}  {_short(r['specific_recipes'])}")
    s = report["summary"]
    order = {lv: i for i, lv in enumerate(LEVELS)}
    fam = ", ".join(f"{k}={v}" for k, v in sorted(s["families_by_level"].items(), key=lambda kv: order[kv[0]]))
    leaf = ", ".join(f"{k}={v}" for k, v in sorted(s["leaf_terms_by_level"].items(), key=lambda kv: order[kv[0]]))
    print(f"\nfamilies ({s['families_total']}): {fam}")
    print(f"leaf terms ({s['leaf_terms_total']}): {leaf}")
    print(f"agent-reachable leaf terms (incl. inherited): {s['leaf_terms_agent_reachable']}/{s['leaf_terms_total']}")
    print(f"recipes: {s['routable_recipes']} routable + {s['pin_only_recipes']} pin-only | "
          f"tagged specs: {s['tagged_specs']} | benchmarks: {s['benchmarks']} {s['benchmark_anchors']}")
    mods = report["modalities"]
    print(f"modalities with a routable recipe: {s['modalities_with_routable_recipe']}/{s['modalities_total']} — "
          + ", ".join(f"{m}={len(v['routable_recipes'])}" for m, v in mods.items()))
    if s["benchmark_anchors"].get("unclassified"):
        print("\nnote: L3*/L4* = scored PASS on a benchmark whose metrics declare no `provenance`;"
              " external vs gads_reference anchoring is unrecorded (approach_docs/033 P0).")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    ap.add_argument("--families", action="store_true", help="table: family rows only")
    ap.add_argument("--save", metavar="PATH", help="also write the JSON report to PATH")
    args = ap.parse_args()

    report = build()
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(report, f, indent=1)
            f.write("\n")
    if args.json:
        print(json.dumps(report, indent=1))
    else:
        print_table(report, families_only=args.families)
        if args.save:
            print(f"\nsaved: {args.save}")


if __name__ == "__main__":
    main()
