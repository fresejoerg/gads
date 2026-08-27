"""Stage a Re-DocRED subset as a knowledge-graph benchmark corpus.

Re-DocRED (MIT, https://github.com/tonytan48/Re-DocRED) re-labels DocRED (MIT,
https://github.com/thunlp/DocRED) to fix its documented false-negative problem. Scoring
against the ORIGINAL would count correct extractions as hallucinations, so this deliberately
uses the revised dev set.

Writes into $GADS_DATASETS_ROOT/redocred:
    corpus_devN.csv         doc_id, title, text
    gold_triplets_devN.csv  doc_id, head, head_type, relation, relation_pid, tail, tail_type
    ontology.json           gads_build_ontology shape, source=user
    PROVENANCE.md           licence + the recall ceiling (see below)

Relation labels come from the Wikidata API (CC0) rather than a repo file, so a moved or
renamed asset upstream cannot silently degrade the labels to bare P-ids.

**Recall ceiling.** DocRED ships tokenised sentences; `text` is a space-join of those
tokens, so a gold `name` like `Wilfried " Willi " Schneider` does not always occur verbatim.
Measured at ~97.6% on the first 100 docs. Since the extraction natives DROP anything not
found verbatim (spans are located, never trusted), that is the benchmark's ceiling — not
extractor error.

Usage:  PYTHONPATH=src uv run python scripts/stage_redocred.py [--n 100]
Network required (host only; the sandbox is offline by design).
"""
import argparse
import csv
import json
import os
import statistics
import time
import urllib.request

DEV_URL = "https://raw.githubusercontent.com/tonytan48/Re-DocRED/main/data/dev_revised.json"
WD_API = ("https://www.wikidata.org/w/api.php?action=wbgetentities&format=json"
          "&languages=en&props=labels|descriptions&ids=")
ETYPE_DESC = {
    "PER": "a person",
    "ORG": "an organisation, company or institution",
    "LOC": "a location, place, city or country",
    "TIME": "a date or time expression",
    "NUM": "a number or quantity",
    "MISC": "a named thing that is none of the above",
}


def _get(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "GADS-benchmark-staging/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def relation_labels(pids):
    """Resolve Wikidata P-ids to English labels/descriptions, batched."""
    out = {}
    for i in range(0, len(pids), 45):
        js = _get(WD_API + "|".join(pids[i:i + 45]), timeout=45)
        for pid, ent in (js.get("entities") or {}).items():
            lab = (ent.get("labels", {}).get("en", {}) or {}).get("value")
            if lab:
                out[pid] = {"label": lab,
                            "description": (ent.get("descriptions", {}).get("en", {}) or {}).get("value", "")}
        time.sleep(0.5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="documents to stage (dev has 500)")
    args = ap.parse_args()

    root = os.getenv("GADS_DATASETS_ROOT", "/home/joergf/datasets")
    out = os.path.join(root, "redocred")
    os.makedirs(out, exist_ok=True)

    print(f"fetching {DEV_URL} ...")
    docs = _get(DEV_URL)
    sub = docs[:args.n]
    print(f"  {len(docs)} dev docs; staging {len(sub)}")

    pids = sorted({l["r"] for d in sub for l in d.get("labels", [])})
    rel = relation_labels(pids)
    print(f"  resolved {len(rel)}/{len(pids)} relation labels from Wikidata")

    def text_of(rec):
        return " ".join(" ".join(s) for s in rec["sents"])

    # Verbatim-occurrence check — this is the benchmark's recall ceiling.
    tot = hit = 0
    for rec in sub:
        t = text_of(rec)
        for cluster in rec["vertexSet"]:
            for m in cluster:
                tot += 1
                hit += m["name"] in t
    ceiling = 100.0 * hit / tot if tot else 0.0
    print(f"  gold mentions verbatim in text: {hit}/{tot} ({ceiling:.1f}%) <- recall ceiling")

    rows, gold = [], []
    for i, rec in enumerate(sub):
        did = f"redocred_{i:04d}"
        rows.append({"doc_id": did, "title": rec["title"], "text": text_of(rec)})
        for l in rec.get("labels", []):
            h, t_ = rec["vertexSet"][l["h"]][0], rec["vertexSet"][l["t"]][0]
            gold.append({"doc_id": did, "head": h["name"], "head_type": h["type"],
                         "relation": rel.get(l["r"], {}).get("label", l["r"]).upper().replace(" ", "_"),
                         "relation_pid": l["r"], "tail": t_["name"], "tail_type": t_["type"]})

    etypes = sorted({m["type"] for r in sub for c in r["vertexSet"] for m in c})
    pid_by_rel = {}
    for g in gold:
        pid_by_rel.setdefault(g["relation"], g["relation_pid"])
    ont = {
        "entity_types": {e: {"description": ETYPE_DESC.get(e, e)} for e in etypes},
        "relation_types": {r: {"description": rel.get(p, {}).get("description") or r.replace("_", " ").lower(),
                               "wikidata_property": p, "domain": [], "range": []}
                           for r, p in sorted(pid_by_rel.items())},
        "source": "user",
    }

    n = len(sub)
    with open(f"{out}/corpus_dev{n}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["doc_id", "title", "text"]); w.writeheader(); w.writerows(rows)
    with open(f"{out}/gold_triplets_dev{n}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["doc_id", "head", "head_type", "relation",
                                          "relation_pid", "tail", "tail_type"])
        w.writeheader(); w.writerows(gold)
    json.dump(ont, open(f"{out}/ontology.json", "w"), indent=2)

    wl = [len(r["text"].split()) for r in rows]
    print(f"\nstaged -> {out}")
    print(f"  {len(rows)} docs | {len(gold)} gold triplets | {len(etypes)} entity types | "
          f"{len(ont['relation_types'])} relation types")
    print(f"  words/doc: median {statistics.median(wl):.0f}, max {max(wl)}")
    print("  NOTE: PROVENANCE.md is maintained by hand — update the ceiling if --n changes.")


if __name__ == "__main__":
    main()
