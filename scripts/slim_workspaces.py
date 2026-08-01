#!/usr/bin/env python3
"""
Reclaim workspace disk by SLIMMING old projects rather than deleting them.

Why slimming and not deletion: a workspace's value is its evidence — the dashboard, the
research report, the exported notebook/script, the metrics and audit JSON. That material is
small. The bulk (dataset copies, pickled models, handover zips, parquet) is large and
reproducible. Stripping the bulk keeps every archive browsable, keeps `workflow_execution.py`
(so kernel rehydration still works — see core/kernel_state.py), and reclaims most of the space.

Two protections, both on by default:
  * **Research evidence** — any project_id cited in research/dial_ledger.jsonl or
    error_ledger.jsonl is skipped entirely. Those runs back published findings; deleting
    their artifacts would break provenance.
  * **Recency** — workspaces newer than --older-than days are untouched.

Symlinked datasets are never removed: they cost no space (the read-only shared root holds
the bytes) and removing them would break replay.

Dry-run by default. Pass --apply to actually delete.

    python scripts/slim_workspaces.py                 # report only
    python scripts/slim_workspaces.py --apply         # reclaim
    python scripts/slim_workspaces.py --older-than 90 --apply
"""
import argparse
import json
import os
import sys
import time

WORKSPACE_ROOT = os.getenv("GADS_WORKSPACE_ROOT",
                           "/home/joergf/projects/MyLocalStack/data/workspaces")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGERS = [os.path.join(REPO_ROOT, "research", "dial_ledger.jsonl"),
           os.path.join(REPO_ROOT, "research", "error_ledger.jsonl")]

# Bulk, reproducible payloads — safe to strip from an aged workspace.
STRIP_EXT = {".csv", ".parquet", ".pkl", ".pickle", ".zip", ".joblib", ".ubj",
             ".pt", ".pth", ".h5", ".hdf5", ".feather", ".arrow", ".npy", ".npz", ".bin"}
# Evidence — always kept regardless of size.
KEEP_NAMES = {"final_dashboard.html", "research_report.md", "metrics.json",
              "workflow_execution.py", "workflow_execution.ipynb", "workflow_spec.md"}


def protected_ids():
    ids = set()
    for path in LEDGERS:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                try:
                    pid = json.loads(line).get("project_id")
                    if pid:
                        ids.add(str(pid))
                except Exception:
                    pass
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--older-than", type=int, default=30, help="age in days (default 30)")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    ap.add_argument("--root", default=WORKSPACE_ROOT)
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"workspace root not found: {args.root}")

    keep_ids = protected_ids()
    cutoff = time.time() - args.older_than * 86400
    tot_free = tot_files = n_slim = 0
    skipped_recent = skipped_protected = 0
    per_ws = []

    for name in sorted(os.listdir(args.root)):
        ws = os.path.join(args.root, name)
        if not os.path.isdir(ws):
            continue
        if name in keep_ids:
            skipped_protected += 1
            continue
        try:
            if os.path.getmtime(ws) > cutoff:
                skipped_recent += 1
                continue
        except OSError:
            continue

        victims, freed = [], 0
        for dirpath, _, files in os.walk(ws):
            for fn in files:
                fp = os.path.join(dirpath, fn)
                if os.path.islink(fp):        # symlinked dataset: no bytes here
                    continue
                if fn in KEEP_NAMES:
                    continue
                if os.path.splitext(fn)[1].lower() not in STRIP_EXT:
                    continue
                try:
                    sz = os.path.getsize(fp)
                except OSError:
                    continue
                victims.append((fp, sz))
                freed += sz
        if not victims:
            continue

        n_slim += 1
        tot_free += freed
        tot_files += len(victims)
        per_ws.append((name, len(victims), freed))

        if args.apply:
            removed = []
            for fp, sz in victims:
                try:
                    os.remove(fp)
                    removed.append({"file": os.path.relpath(fp, ws), "bytes": sz})
                except OSError as e:
                    print(f"  ! could not remove {fp}: {e}")
            # Leave an honest record in the workspace of what was stripped.
            try:
                with open(os.path.join(ws, "slimmed_manifest.json"), "w") as f:
                    json.dump({"slimmed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                               "policy": f"older_than={args.older_than}d",
                               "freed_bytes": freed, "removed": removed}, f, indent=2)
            except OSError:
                pass

    per_ws.sort(key=lambda x: -x[2])
    mode = "APPLIED" if args.apply else "DRY RUN (nothing deleted)"
    print(f"=== slim_workspaces — {mode} ===")
    print(f"root            : {args.root}")
    print(f"age threshold   : > {args.older_than} days")
    print(f"protected (research-cited): {skipped_protected} | skipped (recent): {skipped_recent}")
    print(f"workspaces slimmed        : {n_slim}")
    print(f"files removed             : {tot_files}")
    print(f"space reclaimed           : {tot_free / 1073741824:.2f} GB")
    if per_ws[:10]:
        print("largest:")
        for name, n, freed in per_ws[:10]:
            print(f"   {freed/1048576:9.1f} MB  {n:4d} files  {name}")
    if not args.apply and tot_free:
        print("\nre-run with --apply to reclaim.")


if __name__ == "__main__":
    main()
