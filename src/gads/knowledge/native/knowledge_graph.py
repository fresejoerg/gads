"""
GADS Native Knowledge-Graph Nodes (approach_docs/030)

The deterministic half of knowledge-graph construction: loading and chunking a corpus,
canonicalising entity mentions, materialising a labeled property graph, and auditing the
result. The *extraction* half (which entity/relation types matter, and which triplets a
document actually asserts) stays model-generated — that is judgment, and 019's rule is to
nativize the invariant mechanics and leave the variable work measured.

Two constraints from the sandbox, both measured rather than assumed:

* The only cached model is `sentence-transformers/all-MiniLM-L6-v2`, under
  `HF_HUB_OFFLINE=1`. No NER model, no relation-extraction model, and nothing fetchable.
  Entity resolution is therefore deterministic (normalisation, acronyms, unambiguous
  initials) with MiniLM demoted to a review-suggestion channel — measurements in
  `gads_resolve_entities` show embedding cosine cannot separate "Apple"/"Apple Inc."
  from "North Korea"/"South Korea" at any threshold.
* `nltk` is installed with **no data files** — no `punkt`, no POS tagger. Chunking is
  paragraph/character based with a regex sentence fallback; nothing here may depend on
  nltk corpora.

Provenance is the load-bearing idea. Every chunk carries its document id and character
span, and every edge is expected to carry them through. A triplet that cannot be traced
back to a span in a source document is unauditable, and an unauditable graph is worse than
no graph — it launders a hallucination into something shaped like a fact.

Annotation-free and self-contained (imports inside) so the source injects verbatim into
the sandbox kernel via the preamble.
"""


def gads_load_text_corpus(source, text_col=None, id_col=None, chunk_chars=2400,
                          overlap_chars=300, min_chunk_chars=50, encoding="utf-8"):
    """Load a text corpus and split it into overlapping, provenance-bearing chunks.

    `chunk_chars` default was raised from 1200 to 2400 after measuring the Re-DocRED
    benchmark corpus (research/benchmarks/redocred_kg_v1): documents there run median 1033
    / p90 1701 / max 2376 chars, so 1200 was splitting the majority of documents for no
    reason — a local model's context window is not the binding constraint on a corpus this
    short. 2400 keeps effectively the whole distribution as a single chunk while still
    protecting against a pathologically long outlier document.

    `source` may be a directory of .txt/.md files, a single .txt/.md file, a .jsonl, a
    .csv/.parquet path, or an in-memory DataFrame. For tabular sources `text_col` is
    auto-detected (longest average string column) when not given, and `id_col` defaults to
    the row index.

    Chunks are character windows with overlap, snapped to a paragraph or sentence boundary
    where one falls near the end of the window, so a chunk rarely severs a sentence.
    Deliberately NOT nltk-based: the sandbox ships nltk without its data files.

    Returns {documents, chunks, n_documents, n_chunks, text_col, mean_doc_chars}, where
    `chunks` has columns [chunk_id, doc_id, chunk_index, text, char_start, char_end] —
    char_start/char_end index into the ORIGINAL document, which is what makes downstream
    triplet provenance verifiable.
    """
    import os
    import re
    import json
    import hashlib
    import pandas as pd

    def _read_dir(d):
        rows = []
        for name in sorted(os.listdir(d)):
            if name.lower().endswith((".txt", ".md")):
                p = os.path.join(d, name)
                try:
                    with open(p, encoding=encoding, errors="replace") as fh:
                        rows.append({"doc_id": name, "text": fh.read()})
                except Exception as e:
                    print(f"[gads_load_text_corpus] skipped {name}: {e}")
        return pd.DataFrame(rows)

    # ---- resolve the source into a documents frame -------------------------------
    if isinstance(source, pd.DataFrame):
        docs = source.copy()
    elif isinstance(source, str) and os.path.isdir(source):
        docs = _read_dir(source)
    elif isinstance(source, str) and source.lower().endswith((".txt", ".md")):
        with open(source, encoding=encoding, errors="replace") as fh:
            docs = pd.DataFrame([{"doc_id": os.path.basename(source), "text": fh.read()}])
    elif isinstance(source, str) and source.lower().endswith(".jsonl"):
        rows = []
        with open(source, encoding=encoding, errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
        docs = pd.DataFrame(rows)
    elif isinstance(source, str) and source.lower().endswith(".parquet"):
        docs = pd.read_parquet(source)
    elif isinstance(source, str):
        docs = pd.read_csv(source)
    else:
        raise ValueError("source must be a DataFrame or a path to a dir/.txt/.jsonl/.csv/.parquet")

    if docs is None or len(docs) == 0:
        raise ValueError("corpus is empty — nothing to load")

    # ---- resolve the text column -------------------------------------------------
    if text_col is None:
        if "text" in docs.columns:
            text_col = "text"
        else:
            best, best_len = None, -1.0
            for c in docs.columns:
                if docs[c].dtype == object:
                    try:
                        avg = docs[c].dropna().astype(str).str.len().mean()
                    except Exception:
                        continue
                    if avg is not None and avg > best_len:
                        best, best_len = c, avg
            text_col = best
    if text_col is None or text_col not in docs.columns:
        raise ValueError(f"could not resolve a text column; columns = {list(docs.columns)}")

    if id_col and id_col in docs.columns:
        doc_ids = docs[id_col].astype(str).tolist()
    elif "doc_id" in docs.columns:
        doc_ids = docs["doc_id"].astype(str).tolist()
    else:
        doc_ids = [f"doc_{i}" for i in range(len(docs))]

    texts = docs[text_col].fillna("").astype(str).tolist()

    # ---- chunk with provenance ---------------------------------------------------
    # Snap the window end to a paragraph break, else a sentence end, if one lies in the
    # last 30% of the window. Keeps chunks semantically whole without a tokenizer.
    sent_end = re.compile(r"[.!?]['\"\)\]]?\s")
    chunks = []
    for doc_id, text in zip(doc_ids, texts):
        n = len(text)
        if n == 0:
            continue
        start, idx = 0, 0
        while start < n:
            end = min(start + chunk_chars, n)
            if end < n:
                window_floor = start + int(chunk_chars * 0.7)
                para = text.rfind("\n\n", window_floor, end)
                if para > window_floor:
                    end = para + 2
                else:
                    cands = [m.end() for m in sent_end.finditer(text, window_floor, end)]
                    if cands:
                        end = cands[-1]
            body = text[start:end]
            if len(body.strip()) >= min_chunk_chars or (idx == 0 and body.strip()):
                cid = hashlib.sha1(f"{doc_id}:{start}:{end}".encode()).hexdigest()[:16]
                chunks.append({"chunk_id": cid, "doc_id": doc_id, "chunk_index": idx,
                               "text": body, "char_start": start, "char_end": end})
                idx += 1
            if end >= n:
                break
            start = max(end - overlap_chars, start + 1)

    chunks_df = pd.DataFrame(chunks)
    docs_out = pd.DataFrame({"doc_id": doc_ids, "text": texts})
    mean_chars = float(docs_out["text"].str.len().mean()) if len(docs_out) else 0.0

    print(f"[gads_load_text_corpus] {len(docs_out)} document(s), {len(chunks_df)} chunk(s) "
          f"| text_col='{text_col}' | mean doc {mean_chars:.0f} chars "
          f"| window {chunk_chars}/{overlap_chars}")
    return {"documents": docs_out, "chunks": chunks_df, "n_documents": int(len(docs_out)),
            "n_chunks": int(len(chunks_df)), "text_col": text_col,
            "mean_doc_chars": mean_chars}


def gads_resolve_entities(mentions, surface_col="surface_form", type_col="entity_type",
                          threshold=0.90, model_name="all-MiniLM-L6-v2",
                          max_alias_len=120, allow_embedding_merge=False):
    """Canonicalise entity mentions into graph nodes.

    The step most often skipped, and the one that decides whether the graph is usable:
    "IBM", "I.B.M." and "International Business Machines" must collapse to one node or
    every downstream degree/centrality number is nonsense.

    Blocks strictly by `entity_type` — mentions of different types are NEVER merged, no
    matter how similar the strings. Cross-type merging is the failure that silently fuses
    a PERSON and the ORG named after them.

    Merging uses DETERMINISTIC signals only, as connected components:

    * **normalised exact match** — casefold, strip punctuation and legal suffixes
      (Inc/Corp/Ltd/LLC/PLC/GmbH/SA/AG/NV/BV/Co). Catches "I.B.M." ~ "IBM" and
      "Red Hat" ~ "Red Hat Inc."
    * **acronym ↔ expansion** — "IBM" ~ "International Business Machines"
    * **unambiguous person initials** — "A. Krishna" ~ "Arvind Krishna", but ONLY when
      exactly one expansion matches. "A. Smith" against both "Adam Smith" and "Anna Smith"
      is ambiguous and is left unmerged.

    **Embeddings deliberately do NOT merge by default.** Measured on this MiniLM build:

        "Apple" vs "Apple Inc."              0.735   should merge
        "Bank of America" vs "Bank of England" 0.735  MUST NOT
        "North Korea" vs "South Korea"       0.872   MUST NOT

    The classes are not linearly separable at any threshold — entity names differ by a
    single discriminative token that embeddings under-weight. Any cutoff that merges
    Apple/Apple Inc. also fuses North and South Korea. The errors are asymmetric too: a
    missed merge leaves two visible nodes that can be fixed later, while a wrong merge
    silently fuses two real entities and corrupts every downstream degree, centrality and
    count. So embeddings are demoted to a *suggestion* channel — pairs at or above
    `threshold` are returned as `review_candidates` for a human or a later model pass,
    never merged. Set `allow_embedding_merge=True` to opt in anyway (unsafe for short
    names; reasonable for long descriptive ones).

    Linking is transitive, so A~B and B~C merges all three. That is right for identity and
    is bounded by per-type blocking; `dedup_ratio` is the signal to watch.

    `node_id` is a content hash of (entity_type, canonical_name), so re-running over an
    extended corpus produces stable ids and the graph can be merged rather than rebuilt.

    Returns {nodes, mention_map, n_mentions, n_entities, dedup_ratio, resolution_method}.
    """
    import re
    import hashlib
    import pandas as pd

    if mentions is None or len(mentions) == 0:
        empty = pd.DataFrame(columns=["node_id", "canonical_name", "entity_type",
                                      "aliases", "mention_count"])
        print("[gads_resolve_entities] no mentions supplied — empty node set")
        return {"nodes": empty, "mention_map": pd.DataFrame(), "n_mentions": 0,
                "n_entities": 0, "dedup_ratio": 1.0, "resolution_method": "none"}

    df = pd.DataFrame(mentions).copy()
    if surface_col not in df.columns:
        raise ValueError(f"mentions is missing '{surface_col}'; columns = {list(df.columns)}")
    if type_col not in df.columns:
        df[type_col] = "ENTITY"

    df[surface_col] = df[surface_col].astype(str).str.strip()
    df = df[df[surface_col].str.len() > 0]
    df[type_col] = df[type_col].astype(str).str.strip().str.upper()

    _LEGAL = {"inc", "incorporated", "corp", "corporation", "ltd", "limited", "llc",
              "llp", "plc", "gmbh", "ag", "sa", "nv", "bv", "co", "company", "group",
              "holdings", "the"}

    def _norm(s, strip_legal=True):
        s = re.sub(r"[^\w\s]", " ", str(s).casefold())
        toks = [t for t in s.split() if t]
        if strip_legal:
            stripped = [t for t in toks if t not in _LEGAL]
            if stripped:
                toks = stripped
        return " ".join(toks)

    encoder = None
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(model_name)
        method = "embedding"
    except Exception as e:
        print(f"[gads_resolve_entities] MiniLM unavailable ({type(e).__name__}); "
              f"falling back to normalised exact match")
        method = "exact_norm"

    def _acronym(s):
        words = [w for w in re.split(r"\s+", str(s).strip()) if w]
        return "".join(w[0] for w in words).casefold() if len(words) > 1 else ""

    node_rows, map_rows, review = [], [], []
    for etype, block in df.groupby(type_col, sort=True):
        counts = block[surface_col].value_counts()
        forms = list(counts.index)
        k = len(forms)

        # union-find over the three linking signals
        parent = list(range(k))

        def _find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def _union(i, j):
            ri, rj = _find(i), _find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)

        norms = [_norm(f) for f in forms]
        acros = [_acronym(f) for f in forms]
        # "tight" form drops separators entirely, so a dotted acronym ("I.B.M." -> "ibm")
        # meets its undotted twin. Token-level normalisation alone leaves them as
        # "i b m" vs "ibm" and they never link.
        tights = ["".join(n.split()) for n in norms]
        for i in range(k):
            for j in range(i + 1, k):
                if norms[i] and norms[i] == norms[j]:
                    _union(i, j)                      # punctuation / legal-suffix variants
                elif tights[i] and tights[i] == tights[j]:
                    _union(i, j)                      # dotted / spaced acronym variants
                elif (acros[i] and acros[i] == norms[j]) or (acros[j] and acros[j] == norms[i]):
                    _union(i, j)                      # acronym <-> expansion
                elif (acros[i] and acros[i] == tights[j]) or (acros[j] and acros[j] == tights[i]):
                    _union(i, j)                      # acronym <-> dotted expansion

        # Unambiguous person initials: "A. Krishna" ~ "Arvind Krishna", but only when
        # exactly ONE expansion matches — "A. Smith" against Adam and Anna Smith is
        # genuinely ambiguous and must stay unmerged.
        def _initial_key(n):
            t = n.split()
            return (t[0][0], " ".join(t[1:])) if len(t) >= 2 and len(t[0]) == 1 else None

        for i in range(k):
            ki = _initial_key(norms[i])
            if not ki:
                continue
            cands = [j for j in range(k)
                     if j != i and len(norms[j].split()) >= 2
                     and len(norms[j].split()[0]) > 1
                     and norms[j].split()[0][0] == ki[0]
                     and " ".join(norms[j].split()[1:]) == ki[1]]
            if len(cands) == 1:
                _union(i, cands[0])

        # Embeddings: suggestion channel only (see docstring for the measurement).
        if method == "embedding" and k > 1:
            try:
                import numpy as np
                emb = np.asarray(encoder.encode(forms, normalize_embeddings=True,
                                                show_progress_bar=False))
                sim = emb @ emb.T
                for i in range(k):
                    for j in range(i + 1, k):
                        if sim[i, j] >= threshold and _find(i) != _find(j):
                            if allow_embedding_merge:
                                _union(i, j)
                            else:
                                review.append({"entity_type": etype, "form_a": forms[i],
                                               "form_b": forms[j],
                                               "similarity": round(float(sim[i, j]), 4)})
            except Exception as e:
                print(f"[gads_resolve_entities] embedding pass failed for {etype} "
                      f"({type(e).__name__}); deterministic signals only")

        clusters = {}
        for i, form in enumerate(forms):
            clusters.setdefault(_find(i), []).append(form)

        for members in clusters.values():
            members = sorted(members, key=lambda f: (-int(counts[f]), f))
            canonical = members[0]
            aliases = [m for m in members[1:] if len(m) <= max_alias_len]
            nid = "n_" + hashlib.sha1(f"{etype}|{_norm(canonical)}".encode()).hexdigest()[:16]
            node_rows.append({"node_id": nid, "canonical_name": canonical,
                              "entity_type": etype, "aliases": aliases,
                              "n_aliases": len(aliases),
                              "mention_count": int(sum(int(counts[m]) for m in members))})
            for m in members:
                map_rows.append({"surface_form": m, "entity_type": etype, "node_id": nid,
                                 "canonical_name": canonical})

    nodes = pd.DataFrame(node_rows).sort_values("mention_count", ascending=False)
    mention_map = pd.DataFrame(map_rows)
    n_distinct = int(df[surface_col].nunique())
    n_entities = int(len(nodes))
    ratio = (n_distinct / n_entities) if n_entities else 1.0

    review_df = pd.DataFrame(review).sort_values("similarity", ascending=False) \
        if review else pd.DataFrame(columns=["entity_type", "form_a", "form_b", "similarity"])

    print(f"[gads_resolve_entities] {len(df)} mention(s), {n_distinct} distinct surface "
          f"form(s) -> {n_entities} entities (deterministic"
          + (" + embedding merge" if allow_embedding_merge else "") +
          f") | collapse ratio {ratio:.2f}x")
    if len(review_df):
        print(f"[gads_resolve_entities] {len(review_df)} near-duplicate pair(s) flagged for "
              f"review (>= {threshold} cosine), NOT merged — e.g. "
              f"{review_df.iloc[0]['form_a']!r} ~ {review_df.iloc[0]['form_b']!r}")
    return {"nodes": nodes, "mention_map": mention_map, "n_mentions": int(len(df)),
            "n_entities": n_entities, "dedup_ratio": float(ratio),
            "resolution_method": method, "review_candidates": review_df,
            "n_review_candidates": int(len(review_df))}


def gads_build_lpg(nodes, edges, write_dir=".", graph_name="knowledge_graph",
                   head_col="head_id", tail_col="tail_id", relation_col="relation_type"):
    """Materialise a labeled property graph and persist it.

    There is no graph database in the sandbox and no rdflib, but a `networkx.MultiDiGraph`
    with typed nodes, typed edges and properties on both IS a labeled property graph —
    MultiDiGraph specifically, because two entities can stand in more than one relation and
    collapsing those would lose edges.

    Writes `<name>_nodes.parquet`, `<name>_edges.parquet` (full fidelity, duckdb-queryable)
    and `<name>.graphml` (interchange). GraphML cannot carry list/dict attributes, so those
    are JSON-encoded on export only — the parquet keeps native types.

    Edges whose head or tail is absent from `nodes` are DROPPED rather than silently
    creating phantom nodes, and the count is reported; `gads_audit_graph` flags the same
    condition as an issue.

    Returns {graph, n_nodes, n_edges, n_dropped_edges, nodes_path, edges_path,
    graphml_path, density, n_components}.
    """
    import os
    import json
    import pandas as pd
    import networkx as nx

    nodes = pd.DataFrame(nodes).copy()
    edges = pd.DataFrame(edges).copy() if edges is not None else pd.DataFrame()
    if "node_id" not in nodes.columns:
        raise ValueError("nodes must have a 'node_id' column")

    known = set(nodes["node_id"].astype(str))
    n_dropped = 0
    if len(edges):
        for c in (head_col, tail_col, relation_col):
            if c not in edges.columns:
                raise ValueError(f"edges is missing '{c}'; columns = {list(edges.columns)}")
        before = len(edges)
        edges = edges[edges[head_col].astype(str).isin(known)
                      & edges[tail_col].astype(str).isin(known)]
        n_dropped = before - len(edges)
        if n_dropped:
            print(f"[gads_build_lpg] dropped {n_dropped} edge(s) referencing unknown nodes")

    G = nx.MultiDiGraph()
    for rec in nodes.to_dict("records"):
        nid = str(rec.pop("node_id"))
        G.add_node(nid, **rec)
    for rec in edges.to_dict("records") if len(edges) else []:
        h, t = str(rec.pop(head_col)), str(rec.pop(tail_col))
        rel = str(rec.pop(relation_col))
        G.add_edge(h, t, key=rel, relation_type=rel, **rec)

    os.makedirs(write_dir, exist_ok=True)
    nodes_path = os.path.join(write_dir, f"{graph_name}_nodes.parquet")
    edges_path = os.path.join(write_dir, f"{graph_name}_edges.parquet")
    graphml_path = os.path.join(write_dir, f"{graph_name}.graphml")

    try:
        nodes.to_parquet(nodes_path, index=False)
        (edges if len(edges) else pd.DataFrame(
            columns=[head_col, relation_col, tail_col])).to_parquet(edges_path, index=False)
    except Exception as e:
        nodes_path = nodes_path.replace(".parquet", ".csv")
        edges_path = edges_path.replace(".parquet", ".csv")
        print(f"[gads_build_lpg] parquet unavailable ({type(e).__name__}); wrote CSV")
        nodes.to_csv(nodes_path, index=False)
        edges.to_csv(edges_path, index=False)

    try:
        H = nx.MultiDiGraph()
        for n, d in G.nodes(data=True):
            H.add_node(n, **{k: (json.dumps(v) if isinstance(v, (list, dict)) else
                                 ("" if v is None else v)) for k, v in d.items()})
        for u, v, k, d in G.edges(keys=True, data=True):
            H.add_edge(u, v, key=k, **{kk: (json.dumps(vv) if isinstance(vv, (list, dict))
                                            else ("" if vv is None else vv))
                                       for kk, vv in d.items()})
        nx.write_graphml(H, graphml_path)
    except Exception as e:
        graphml_path = None
        print(f"[gads_build_lpg] GraphML export skipped: {type(e).__name__}: {e}")

    n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()
    density = nx.density(G) if n_nodes > 1 else 0.0
    try:
        n_components = nx.number_weakly_connected_components(G)
    except Exception:
        n_components = None

    print(f"[gads_build_lpg] {n_nodes} node(s), {n_edges} edge(s) | density {density:.4f} "
          f"| {n_components} weakly-connected component(s)")
    print(f"[gads_build_lpg] wrote {os.path.basename(nodes_path)}, "
          f"{os.path.basename(edges_path)}"
          + (f", {os.path.basename(graphml_path)}" if graphml_path else ""))
    return {"graph": G, "n_nodes": int(n_nodes), "n_edges": int(n_edges),
            "n_dropped_edges": int(n_dropped), "nodes_path": nodes_path,
            "edges_path": edges_path, "graphml_path": graphml_path,
            "density": float(density), "n_components": n_components}


def gads_audit_graph(nodes, edges, ontology=None, write_path="graph_checks.json",
                     head_col="head_id", tail_col="tail_id", relation_col="relation_type",
                     min_provenance_frac=0.99, emit_insights=True):
    """Methodological gate for a constructed knowledge graph.

    Mirrors `gads_audit_model_choice`: turns "is this graph trustworthy" into checks that
    run regardless of what the Coder wrote, writes `graph_checks.json`, and returns the
    findings rather than raising.

    Issues (a graph with any of these should not be reported as fact):
      * `dangling_reference`  — an edge points at a node that does not exist
      * `missing_provenance`  — edges without doc_id / char span / confidence. The
        headline check: an untraceable triplet launders a hallucination into a fact
      * `ontology_violation`  — relation used between types its domain/range forbids
      * `degenerate_confidence` — every edge has the same confidence (typically all 1.0),
        which means the extractor is not calibrated and the scores carry no information
      * `contradiction`       — mutually exclusive relations asserted for one pair

    Tips (worth knowing, not disqualifying): self-loops, duplicate triplets, isolated
    nodes, and a collapse ratio of ~1.0 (entity resolution did nothing, so aliases are
    probably still separate nodes).

    `ontology`, when supplied, is {"relation_types": {REL: {"domain": [...], "range": [...],
    "mutually_exclusive_with": [...]}}}. Absent, conformance checks are skipped and
    reported as not applicable rather than silently passing.

    Returns {issues, tips, passed, not_applicable, n_issues, n_tips, stats, checks_path}.
    """
    import json
    import pandas as pd

    result = {"issues": [], "tips": [], "passed": [], "not_applicable": [],
              "n_issues": 0, "n_tips": 0, "stats": {}, "checks_path": write_path}

    def _issue(code, title, detail):
        result["issues"].append({"code": code, "title": title, "detail": detail})

    def _tip(code, title, detail):
        result["tips"].append({"code": code, "title": title, "detail": detail})

    try:
        nodes = pd.DataFrame(nodes).copy()
        edges = pd.DataFrame(edges).copy() if edges is not None else pd.DataFrame()
        n_nodes, n_edges = len(nodes), len(edges)
        result["stats"] = {"n_nodes": int(n_nodes), "n_edges": int(n_edges)}

        if n_edges == 0:
            _issue("empty_graph", "Graph has no edges",
                   "Extraction produced no triplets; nothing downstream is meaningful.")
            result["n_issues"] = len(result["issues"])
            with open(write_path, "w") as fh:
                json.dump(result, fh, indent=2, default=str)
            print("[gads_audit_graph] 1 issue: graph has no edges")
            return result

        known = set(nodes["node_id"].astype(str)) if "node_id" in nodes.columns else set()

        # 1. dangling references
        if known:
            bad = edges[~edges[head_col].astype(str).isin(known)
                        | ~edges[tail_col].astype(str).isin(known)]
            if len(bad):
                _issue("dangling_reference", "Edges reference unknown nodes",
                       f"{len(bad)} of {n_edges} edge(s) point at a node_id absent from "
                       f"the node table.")
            else:
                result["passed"].append("dangling_reference")
        else:
            result["not_applicable"].append("dangling_reference")

        # 2. provenance — the headline check
        prov_cols = [c for c in ("doc_id", "chunk_id") if c in edges.columns]
        span_cols = [c for c in ("char_start", "char_end") if c in edges.columns]
        if not prov_cols:
            _issue("missing_provenance", "Edges carry no document provenance",
                   "No doc_id/chunk_id column. Every triplet must be traceable to the "
                   "text that asserts it; without it the graph cannot be audited.")
        else:
            traced = edges[prov_cols[0]].notna().mean()
            result["stats"]["provenance_frac"] = float(traced)
            if traced < min_provenance_frac:
                _issue("missing_provenance", "Some edges are untraceable",
                       f"only {traced:.1%} of edges carry {prov_cols[0]} "
                       f"(threshold {min_provenance_frac:.0%}).")
            else:
                result["passed"].append("missing_provenance")
            if not span_cols:
                _tip("missing_span", "No character spans on edges",
                     "doc_id is present but char_start/char_end are not; spans make a "
                     "claim checkable against the exact sentence.")

        # 3. confidence calibration
        if "confidence" in edges.columns:
            conf = pd.to_numeric(edges["confidence"], errors="coerce").dropna()
            if len(conf) and conf.nunique() == 1:
                _issue("degenerate_confidence", "All edges share one confidence value",
                       f"every edge has confidence={conf.iloc[0]}; the extractor is not "
                       f"calibrated and the score carries no information.")
            elif len(conf):
                result["passed"].append("degenerate_confidence")
                result["stats"]["confidence_mean"] = float(conf.mean())
                result["stats"]["confidence_min"] = float(conf.min())
        else:
            _tip("no_confidence", "Edges carry no confidence score",
                 "Without it, low-certainty extractions cannot be filtered or ranked.")

        # 4. ontology conformance
        rel_defs = (ontology or {}).get("relation_types") or {}
        if rel_defs and "entity_type" in nodes.columns and known:
            types = dict(zip(nodes["node_id"].astype(str),
                             nodes["entity_type"].astype(str)))
            violations = []
            for rec in edges.to_dict("records"):
                spec = rel_defs.get(str(rec.get(relation_col)))
                if not spec:
                    continue
                ht, tt = types.get(str(rec.get(head_col))), types.get(str(rec.get(tail_col)))
                dom, rng = spec.get("domain") or [], spec.get("range") or []
                if (dom and ht not in dom) or (rng and tt not in rng):
                    violations.append(f"{ht} -{rec.get(relation_col)}-> {tt}")
            if violations:
                uniq = sorted(set(violations))[:5]
                _issue("ontology_violation", "Relations used outside their domain/range",
                       f"{len(violations)} violating edge(s); e.g. {uniq}")
            else:
                result["passed"].append("ontology_violation")
        else:
            result["not_applicable"].append("ontology_violation")

        # 5. contradictions
        excl = {r: set(s.get("mutually_exclusive_with") or [])
                for r, s in rel_defs.items() if s.get("mutually_exclusive_with")}
        if excl:
            pair_rels = {}
            for rec in edges.to_dict("records"):
                pair_rels.setdefault((str(rec.get(head_col)), str(rec.get(tail_col))),
                                     set()).add(str(rec.get(relation_col)))
            clashes = []
            for pair, rels in pair_rels.items():
                for r in rels:
                    if excl.get(r, set()) & rels:
                        clashes.append(f"{pair}: {sorted(rels)}")
                        break
            if clashes:
                _issue("contradiction", "Mutually exclusive relations on the same pair",
                       f"{len(clashes)} pair(s); e.g. {clashes[:3]}")
            else:
                result["passed"].append("contradiction")
        else:
            result["not_applicable"].append("contradiction")

        # 6. structural tips
        selfloops = edges[edges[head_col].astype(str) == edges[tail_col].astype(str)]
        if len(selfloops):
            _tip("self_loops", "Self-referential edges",
                 f"{len(selfloops)} edge(s) link a node to itself.")
        dup = edges.duplicated(subset=[head_col, relation_col, tail_col]).sum()
        if dup:
            _tip("duplicate_triplets", "Repeated triplets",
                 f"{int(dup)} duplicate (head, relation, tail) row(s); consider "
                 f"aggregating into an edge weight or evidence count.")
        if known:
            linked = set(edges[head_col].astype(str)) | set(edges[tail_col].astype(str))
            isolated = len(known - linked)
            result["stats"]["n_isolated_nodes"] = int(isolated)
            if isolated:
                _tip("isolated_nodes", "Entities with no relations",
                     f"{isolated} of {n_nodes} node(s) participate in no edge.")

        result["n_issues"] = len(result["issues"])
        result["n_tips"] = len(result["tips"])

        try:
            with open(write_path, "w") as fh:
                json.dump(result, fh, indent=2, default=str)
        except Exception as e:
            print(f"[gads_audit_graph] could not write {write_path}: {e}")

        if result["issues"]:
            print(f"[gads_audit_graph] {result['n_issues']} ISSUE(S):")
            for i in result["issues"]:
                print(f"  - {i['title']}: {i['detail']}")
        else:
            print(f"[gads_audit_graph] no issues across {n_nodes} nodes / {n_edges} edges")
        for t in result["tips"]:
            print(f"  ~ {t['title']}: {t['detail']}")

        if emit_insights:
            emitter = globals().get("gads_emit_insight")
            if callable(emitter):
                for i in result["issues"]:
                    try:
                        emitter("knowledge_graph", f"Graph audit: {i['title']}", i["detail"])
                    except Exception:
                        pass
    except Exception as e:
        # Fail-open: a diagnostic must never be what fails the task.
        print(f"[gads_audit_graph] audit error ({type(e).__name__}: {e}); continuing")
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def gads_build_ontology(entity_types=None, relation_types=None, source=None,
                        write_path="ontology.json"):
    """Normalise an ontology into the canonical shape, from user input or induction.

    Three provenances, all landing in the same structure so nothing downstream cares:

    * **user** — the analyst already knows the schema (biomedical DRUG/GENE/DISEASE,
      finance ORG/INSTRUMENT). The common enterprise case, and by far the most reliable:
      extraction constrained to a known type set beats extraction that also has to guess
      the set.
    * **induced** — proposed by the model from a corpus sample.
    * **hybrid** — user types are authoritative, model additions are marked `added_by`
      so a later audit can tell which came from where.

    Input is accepted loosely, because a user should not have to learn a schema to say
    "PERSON, ORG". `entity_types` may be a list of names, {name: description}, or
    {name: {description, examples}}. `relation_types` may be a list of names, a list of
    "HEAD REL TAIL" strings, {name: description}, or full
    {name: {description, domain, range, examples, mutually_exclusive_with}}.

    Normalisation is deliberate and reported: names are upper-snake-cased, so "works for"
    and "Works For" cannot become two relations. Domain/range default to open (any type)
    rather than to a guess — an unconstrained relation is honest, a wrongly-guessed one
    silently rejects valid triplets in `gads_audit_graph`.

    Returns {entity_types, relation_types, source, n_entity_types, n_relation_types,
    warnings, ontology_path} — the same shape `gads_audit_graph(ontology=...)` consumes.
    """
    import re
    import json

    warnings = []

    def _canon(name):
        s = re.sub(r"[^\w\s]", " ", str(name)).strip()
        s = re.sub(r"\s+", "_", s)
        return s.upper()

    def _norm_types(spec, kind):
        out = {}
        if spec is None:
            return out
        if isinstance(spec, (list, tuple, set)):
            items = {}
            for it in spec:
                if isinstance(it, dict):
                    nm = it.get("name") or it.get("type") or it.get("id")
                    if nm:
                        items[nm] = {k: v for k, v in it.items()
                                     if k not in ("name", "type", "id")}
                else:
                    items[str(it)] = {}
            spec = items
        if not isinstance(spec, dict):
            raise ValueError(f"{kind} must be a list or dict, got {type(spec).__name__}")
        for raw, body in spec.items():
            name = _canon(raw)
            if not name:
                warnings.append(f"dropped unnameable {kind} entry: {raw!r}")
                continue
            if name != str(raw):
                warnings.append(f"{kind} {raw!r} normalised to {name!r}")
            if isinstance(body, str):
                body = {"description": body}
            elif not isinstance(body, dict):
                body = {}
            if name in out:
                warnings.append(f"duplicate {kind} {name!r} after normalisation — merged")
                out[name].update({k: v for k, v in body.items() if v})
            else:
                out[name] = dict(body)
        return out

    ents = _norm_types(entity_types, "entity_type")

    # Relations given as "ORG ACQUIRED ORG" carry their own domain/range.
    rel_spec = relation_types
    if isinstance(rel_spec, (list, tuple, set)):
        expanded = {}
        for it in rel_spec:
            if isinstance(it, str) and len(it.split()) == 3:
                h, r, t = it.split()
                expanded[r] = {"domain": [_canon(h)], "range": [_canon(t)]}
            else:
                expanded[it if not isinstance(it, dict) else
                         (it.get("name") or it.get("type") or "REL")] = (
                    it if isinstance(it, dict) else {})
        rel_spec = expanded
    rels = _norm_types(rel_spec, "relation_type")

    for name, body in rels.items():
        for key in ("domain", "range"):
            val = body.get(key)
            if val is None:
                body[key] = []          # open by default — see docstring
            elif isinstance(val, str):
                body[key] = [_canon(val)]
            else:
                body[key] = [_canon(v) for v in val]
            unknown = [t for t in body[key] if ents and t not in ents]
            if unknown:
                warnings.append(
                    f"relation {name!r} {key} references undeclared entity type(s) "
                    f"{unknown} — they will be added")
                for t in unknown:
                    ents.setdefault(t, {"description": f"(implied by {name}.{key})"})
        mx = body.get("mutually_exclusive_with")
        body["mutually_exclusive_with"] = ([_canon(m) for m in mx] if mx else [])

    if not ents:
        warnings.append("no entity types declared — extraction will be unconstrained")
    if not rels:
        warnings.append("no relation types declared — triplet extraction will be unconstrained")

    ontology = {"entity_types": ents, "relation_types": rels,
                "source": source or ("user" if (entity_types or relation_types) else "empty"),
                "n_entity_types": len(ents), "n_relation_types": len(rels),
                "warnings": warnings, "ontology_path": write_path}
    try:
        import os as _os
        _d = _os.path.dirname(write_path)
        if _d:
            _os.makedirs(_d, exist_ok=True)
        with open(write_path, "w") as fh:
            json.dump({k: v for k, v in ontology.items() if k != "ontology_path"},
                      fh, indent=2, default=str)
    except Exception as e:
        print(f"[gads_build_ontology] could not write {write_path}: {e}")

    print(f"[gads_build_ontology] source={ontology['source']} | {len(ents)} entity type(s): "
          f"{sorted(ents)[:8]}{'...' if len(ents) > 8 else ''}")
    print(f"[gads_build_ontology] {len(rels)} relation type(s): "
          f"{sorted(rels)[:8]}{'...' if len(rels) > 8 else ''}")
    for w in warnings:
        print(f"  ~ {w}")
    return ontology


def gads_extract_entities(chunks, ontology, max_chunks=200, max_chars=6000,
                          model="local_model", temperature=0.0, timeout=120.0,
                          max_output_tokens=3000, write_path="mentions.parquet"):
    """Extract typed entity mentions from chunks, constrained to the ontology's types.

    Calls the local model through the sandbox-scoped LiteLLM key
    (`GADS_SANDBOX_LLM_KEY`, `models: [local_model]` — cloud providers are refused, so the
    corpus cannot leave this machine). Raises if that env is absent rather than silently
    producing nothing: a missing credential is a setup fault, not an empty result.

    **Spans are computed, never trusted.** The model returns surface forms; this locates
    each one in the chunk with `str.find` and converts to a document-absolute offset via
    the chunk's `char_start`. Model-reported offsets are routinely wrong by a few
    characters, and a wrong offset is worse than no offset — it makes an unverifiable claim
    look verifiable. A form the model returns that does not occur in the chunk is DROPPED
    and counted in `n_hallucinated`, because it was not in the text.

    Types outside the ontology are dropped and counted in `n_off_ontology`. `max_chunks`
    and `max_chars` are hard budget caps — generated code can loop, and the key's rpm limit
    should not be the only thing standing between a 10k-document corpus and an afternoon.

    `max_output_tokens` bounds each completion. It was a hardcoded 1500 — measured against
    the Re-DocRED benchmark corpus that silently truncated the response for the densest
    documents (13/100 docs' gold exceeded it at ~28 tokens/item). Raised to a 3000 default;
    still a hard cap generated code should widen deliberately for denser domains, not a
    guess to trust blindly.

    Returns {mentions, n_mentions, n_chunks_processed, n_hallucinated, n_off_ontology,
    n_failed_chunks, mentions_path}.
    """
    import os
    import re
    import json
    import pandas as pd

    base = os.environ.get("GADS_SANDBOX_LLM_BASE_URL")
    key = os.environ.get("GADS_SANDBOX_LLM_KEY")
    if not base or not key:
        raise RuntimeError(
            "GADS_SANDBOX_LLM_BASE_URL / GADS_SANDBOX_LLM_KEY are not set in the sandbox. "
            "Extraction needs the local-model-scoped LiteLLM key — see approach_docs/030 §2a.")
    from openai import OpenAI
    client = OpenAI(base_url=base, api_key=key, timeout=timeout, max_retries=1)

    chunks = pd.DataFrame(chunks)
    allowed = set((ontology or {}).get("entity_types") or {})
    type_help = "\n".join(
        f"- {t}: {(b or {}).get('description') or 'no description'}"
        for t, b in ((ontology or {}).get("entity_types") or {}).items()) or "- ENTITY: any"

    budget = chunks.head(max_chunks)
    rows, n_hall, n_off, n_failed, used = [], 0, 0, 0, 0

    for rec in budget.to_dict("records"):
        text = str(rec.get("text") or "")[:max_chars]
        if not text.strip():
            continue
        prompt = (
            "Extract named entities from the TEXT. Use ONLY these types:\n"
            f"{type_help}\n\n"
            "Return ONLY a JSON array, no prose, no code fences. Each element:\n"
            '{"surface_form": "<exact substring copied from TEXT>", "entity_type": "<TYPE>"}\n'
            "The surface_form MUST appear verbatim in TEXT. Return [] if there are none.\n\n"
            f"TEXT:\n{text}")
        try:
            resp = client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_output_tokens,
                messages=[{"role": "user", "content": prompt}])
            raw = (resp.choices[0].message.content or "").strip()
            used += 1
        except Exception as e:
            n_failed += 1
            print(f"[gads_extract_entities] chunk {rec.get('chunk_id')} failed: "
                  f"{type(e).__name__}: {str(e)[:90]}")
            continue

        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            n_failed += 1
            continue
        try:
            items = json.loads(m.group(0))
        except Exception:
            n_failed += 1
            continue

        for it in items if isinstance(items, list) else []:
            if not isinstance(it, dict):
                continue
            form = str(it.get("surface_form") or "").strip()
            etype = str(it.get("entity_type") or "").strip().upper()
            if not form:
                continue
            if allowed and etype not in allowed:
                n_off += 1
                continue
            pos = text.find(form)
            if pos < 0:                       # not in the text — do not invent a span
                n_hall += 1
                continue
            base_off = int(rec.get("char_start") or 0)
            rows.append({"surface_form": form, "entity_type": etype or "ENTITY",
                         "doc_id": rec.get("doc_id"), "chunk_id": rec.get("chunk_id"),
                         "char_start": base_off + pos,
                         "char_end": base_off + pos + len(form)})

    mentions = pd.DataFrame(rows)
    if len(mentions):
        mentions = mentions.drop_duplicates(
            subset=["surface_form", "entity_type", "doc_id", "char_start"])
        # Persisting is a convenience; the caller already has the frame. A bad path must
        # never destroy an extraction that cost real model calls.
        try:
            _d = os.path.dirname(write_path)
            if _d:
                os.makedirs(_d, exist_ok=True)
            try:
                mentions.to_parquet(write_path, index=False)
            except Exception:
                write_path = write_path.replace(".parquet", ".csv")
                mentions.to_csv(write_path, index=False)
        except Exception as e:
            print(f"[gads_extract_entities] could not persist to {write_path}: "
                  f"{type(e).__name__}; returning in-memory only")
            write_path = None

    skipped = max(0, len(chunks) - len(budget))
    print(f"[gads_extract_entities] {len(mentions)} mention(s) from {used}/{len(chunks)} "
          f"chunk(s)" + (f" (budget capped, {skipped} skipped)" if skipped else "")
          + f" | dropped: {n_hall} not-in-text, {n_off} off-ontology, {n_failed} failed call(s)")
    return {"mentions": mentions, "n_mentions": int(len(mentions)),
            "n_chunks_processed": int(used), "n_hallucinated": int(n_hall),
            "n_off_ontology": int(n_off), "n_failed_chunks": int(n_failed),
            "mentions_path": write_path}


def gads_extract_triplets(chunks, ontology, max_chunks=200, max_chars=6000,
                          model="local_model", temperature=0.0, timeout=120.0,
                          max_output_tokens=3000, write_path="triplets.parquet"):
    """Extract (head, relation, tail) triplets from chunks, constrained to the ontology.

    Same contract as `gads_extract_entities`: local model only, budget-capped, and spans
    are LOCATED rather than believed — head and tail must both occur verbatim in the chunk
    or the triplet is dropped as unsupported. The edge's `char_start`/`char_end` bracket
    the head and tail occurrences, so the span quotes the text that actually asserts the
    relation.

    Relations outside the ontology are dropped (`n_off_ontology`). Domain/range are NOT
    enforced here — `gads_audit_graph` owns that check, and it is more useful as a reported
    finding on the assembled graph than as a silent filter that hides what the model did.

    `confidence` is the model's own estimate, clamped to [0,1] and defaulting to 0.5. It is
    weak evidence by construction; the audit's degenerate-confidence check exists precisely
    because a model that returns 1.0 for everything is telling you nothing.

    `max_output_tokens` bounds each completion. It was a hardcoded 1500 — measured against
    the Re-DocRED benchmark corpus (median 33 / max 92 gold triplets per doc, ~28 tokens
    each) that would have silently truncated the response on the densest 13/100 documents.
    Raised to a 3000 default; still a hard cap generated code should widen deliberately for
    denser domains, not a guess to trust blindly.

    Returns {triplets, n_triplets, n_chunks_processed, n_unsupported, n_off_ontology,
    n_failed_chunks, triplets_path}.
    """
    import os
    import re
    import json
    import pandas as pd

    base = os.environ.get("GADS_SANDBOX_LLM_BASE_URL")
    key = os.environ.get("GADS_SANDBOX_LLM_KEY")
    if not base or not key:
        raise RuntimeError(
            "GADS_SANDBOX_LLM_BASE_URL / GADS_SANDBOX_LLM_KEY are not set in the sandbox. "
            "Extraction needs the local-model-scoped LiteLLM key — see approach_docs/030 §2a.")
    from openai import OpenAI
    client = OpenAI(base_url=base, api_key=key, timeout=timeout, max_retries=1)

    chunks = pd.DataFrame(chunks)
    rel_defs = (ontology or {}).get("relation_types") or {}
    allowed = set(rel_defs)
    rel_help = "\n".join(
        f"- {r}: {(b or {}).get('description') or 'no description'}"
        + (f" [{'/'.join(b.get('domain') or [])} -> {'/'.join(b.get('range') or [])}]"
           if (b or {}).get("domain") or (b or {}).get("range") else "")
        for r, b in rel_defs.items()) or "- RELATED_TO: any relation"

    budget = chunks.head(max_chunks)
    rows, n_unsup, n_off, n_failed, used = [], 0, 0, 0, 0

    for rec in budget.to_dict("records"):
        text = str(rec.get("text") or "")[:max_chars]
        if not text.strip():
            continue
        prompt = (
            "Extract relationship triplets from the TEXT. Use ONLY these relation types:\n"
            f"{rel_help}\n\n"
            "Return ONLY a JSON array, no prose, no code fences. Each element:\n"
            '{"head": "<exact substring from TEXT>", "relation": "<TYPE>", '
            '"tail": "<exact substring from TEXT>", "confidence": <0.0-1.0>}\n'
            "head and tail MUST appear verbatim in TEXT. Only state relations the TEXT "
            "actually asserts — do not infer from world knowledge. Return [] if none.\n\n"
            f"TEXT:\n{text}")
        try:
            resp = client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=max_output_tokens,
                messages=[{"role": "user", "content": prompt}])
            raw = (resp.choices[0].message.content or "").strip()
            used += 1
        except Exception as e:
            n_failed += 1
            print(f"[gads_extract_triplets] chunk {rec.get('chunk_id')} failed: "
                  f"{type(e).__name__}: {str(e)[:90]}")
            continue

        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            n_failed += 1
            continue
        try:
            items = json.loads(m.group(0))
        except Exception:
            n_failed += 1
            continue

        base_off = int(rec.get("char_start") or 0)
        for it in items if isinstance(items, list) else []:
            if not isinstance(it, dict):
                continue
            head = str(it.get("head") or "").strip()
            tail = str(it.get("tail") or "").strip()
            rel = str(it.get("relation") or "").strip().upper().replace(" ", "_")
            if not head or not tail or not rel:
                continue
            if allowed and rel not in allowed:
                n_off += 1
                continue
            hp, tp = text.find(head), text.find(tail)
            if hp < 0 or tp < 0:              # the text does not contain what was claimed
                n_unsup += 1
                continue
            try:
                conf = float(it.get("confidence", 0.5))
            except Exception:
                conf = 0.5
            conf = min(1.0, max(0.0, conf))
            lo = min(hp, tp)
            hi = max(hp + len(head), tp + len(tail))
            rows.append({"head_surface": head, "relation_type": rel, "tail_surface": tail,
                         "confidence": conf, "doc_id": rec.get("doc_id"),
                         "chunk_id": rec.get("chunk_id"),
                         "char_start": base_off + lo, "char_end": base_off + hi,
                         "evidence": text[lo:hi][:300]})

    triplets = pd.DataFrame(rows)
    if len(triplets):
        triplets = triplets.drop_duplicates(
            subset=["head_surface", "relation_type", "tail_surface", "doc_id", "char_start"])
        try:
            _d = os.path.dirname(write_path)
            if _d:
                os.makedirs(_d, exist_ok=True)
            try:
                triplets.to_parquet(write_path, index=False)
            except Exception:
                write_path = write_path.replace(".parquet", ".csv")
                triplets.to_csv(write_path, index=False)
        except Exception as e:
            print(f"[gads_extract_triplets] could not persist to {write_path}: "
                  f"{type(e).__name__}; returning in-memory only")
            write_path = None

    skipped = max(0, len(chunks) - len(budget))
    print(f"[gads_extract_triplets] {len(triplets)} triplet(s) from {used}/{len(chunks)} "
          f"chunk(s)" + (f" (budget capped, {skipped} skipped)" if skipped else "")
          + f" | dropped: {n_unsup} unsupported-by-text, {n_off} off-ontology, "
          f"{n_failed} failed call(s)")
    return {"triplets": triplets, "n_triplets": int(len(triplets)),
            "n_chunks_processed": int(used), "n_unsupported": int(n_unsup),
            "n_off_ontology": int(n_off), "n_failed_chunks": int(n_failed),
            "triplets_path": write_path}
