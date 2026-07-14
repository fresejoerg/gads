"""Render the delegation-dial evidence grid: pass rates per (rung × engine mode).

Usage:
    uv run python scripts/dial_heatmap.py                      # markdown table
    uv run python scripts/dial_heatmap.py --png heatmap.png    # + matplotlib figure

Reads research/dial_ledger.jsonl (one record per completed run, written by the
workflow — see core/dial.py). Cells show pass/total; a cell is only as trustworthy as
its n, and runs are observational (specs are not randomly assigned to cells).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

RUNGS = ["D0", "D1", "D2", "D3", "D4", "D5"]
MODES = ["local", "hybrid", "cloud", "cloud_pinned"]


def load(path: Path):
    cells = defaultdict(lambda: [0, 0])  # (rung, mode) -> [pass, total]
    if not path.exists():
        return cells
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rung, mode = r.get("rung"), r.get("routing_mode")
        if rung is None or mode is None:
            continue
        cells[(rung, mode)][1] += 1
        if r.get("outcome") == "pass":
            cells[(rung, mode)][0] += 1
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ledger", default="research/dial_ledger.jsonl")
    ap.add_argument("--png", default=None, help="also write a matplotlib heatmap")
    args = ap.parse_args()

    cells = load(Path(args.ledger))
    total = sum(t for _, t in cells.values())
    print(f"Delegation-dial evidence grid — {total} run(s) in {args.ledger}\n")
    header = "| rung |" + "".join(f" {m} |" for m in MODES)
    print(header)
    print("|---|" + "---|" * len(MODES))
    for rung in reversed(RUNGS):  # most prescribed on top
        row = f"| {rung} |"
        for mode in MODES:
            p, t = cells.get((rung, mode), (0, 0))
            row += f" {p}/{t} ({p/t:.0%}) |" if t else " — |"
        print(row)
    print("\nD5=mechanized … D0=full delegation. Cells: pass/total. "
          "pass = approved synthesis AND zero failed tasks.")

    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        grid = np.full((len(RUNGS), len(MODES)), np.nan)
        for (rung, mode), (p, t) in cells.items():
            if rung in RUNGS and mode in MODES and t:
                grid[RUNGS.index(rung), MODES.index(mode)] = p / t
        fig, ax = plt.subplots(figsize=(7, 5))
        # Sequential single hue: pass rate is a magnitude, and red-green scales are
        # unreadable for the most common color-vision deficiency.
        im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(MODES)), MODES)
        ax.set_yticks(range(len(RUNGS)), RUNGS)
        ax.invert_yaxis()
        for (rung, mode), (p, t) in cells.items():
            if rung in RUNGS and mode in MODES and t:
                ax.text(MODES.index(mode), RUNGS.index(rung), f"{p}/{t}",
                        ha="center", va="center", fontsize=11)
        ax.set_title("Pass rate by delegation rung × engine mode")
        fig.colorbar(im, label="pass rate")
        plt.tight_layout()
        plt.savefig(args.png, dpi=120)
        print(f"\nWrote {args.png}")


if __name__ == "__main__":
    main()
