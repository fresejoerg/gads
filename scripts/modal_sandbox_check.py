"""Runs INSIDE a sandbox built from MyLocalStack/sandbox/Dockerfile (HOSTED-3b).

Exercises what an import check misses: Chromium under the runtime (kaleido), the baked
offline embedding model, the compiled packages, a torch-backed AutoGluon timeseries fit, and
every GADS native preamble. Prints one `CHECK <name> ok|FAIL <detail>` line per check.
Preambles are read from /tmp/gads_preambles.json ({name: source}) when present.
"""
import json
import os
import time
import traceback


def check(name):
    def wrap(fn):
        t0 = time.time()
        try:
            detail = fn() or ""
            print(f"CHECK {name} ok {time.time() - t0:.1f}s {detail}", flush=True)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc().strip().splitlines()[-3:]
            print(f"CHECK {name} FAIL {type(exc).__name__}: {exc} | {' / '.join(tb)}", flush=True)
        return fn
    return wrap


@check("identity")
def _():
    st = os.stat("/app/app.py")
    return f"uid={os.getuid()} app.py uid={st.st_uid} mode={oct(st.st_mode & 0o777)} cwd={os.getcwd()}"


@check("torch_cpu")
def _():
    import torch
    assert "+cpu" in torch.__version__, torch.__version__
    return torch.__version__


@check("kaleido_png")
def _():
    import plotly.graph_objects as go
    png = go.Figure(go.Scatter(y=[1, 3, 2])).to_image(format="png")
    assert png[:4] == b"\x89PNG", png[:8]
    return f"{len(png)} bytes"


@check("embedding_offline")
def _():
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    from sentence_transformers import SentenceTransformer
    v = SentenceTransformer("all-MiniLM-L6-v2").encode(["hello world"])
    return f"dim={v.shape[1]}"


@check("implicit_als")
def _():
    import numpy as np
    import scipy.sparse as sp
    from implicit.als import AlternatingLeastSquares
    m = sp.random(50, 40, density=0.1, format="csr", random_state=0)
    model = AlternatingLeastSquares(factors=4, iterations=2, random_state=0)
    model.fit(m, show_progress=False)
    return f"user_factors={model.user_factors.shape}"


@check("lightgbm")
def _():
    import lightgbm as lgb
    import numpy as np
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(200, 5)), rng.integers(0, 2, 200)
    lgb.LGBMClassifier(n_estimators=10, verbose=-1).fit(X, y)


@check("autogluon_ts_torch")
def _():
    import numpy as np
    import pandas as pd
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    idx = pd.date_range("2000-01-01", periods=60, freq="MS")
    df = pd.DataFrame({"item_id": "a", "timestamp": idx, "target": np.sin(np.arange(60)) + 10})
    tsdf = TimeSeriesDataFrame.from_data_frame(df)
    p = TimeSeriesPredictor(prediction_length=6, path="/tmp/ag_ts", verbosity=0).fit(
        tsdf, hyperparameters={"DeepAR": {"max_epochs": 1, "num_batches_per_epoch": 2}}, time_limit=120)
    return f"models={p.model_names()}"


@check("native_preambles")
def _():
    path = "/tmp/gads_preambles.json"
    if not os.path.exists(path):
        return "skipped (no preambles file)"
    preambles = json.load(open(path))
    failed = []
    for name, src in preambles.items():
        try:
            exec(compile(src, f"<{name}>", "exec"), {"__name__": "__gads__"})
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failed, failed
    return f"{len(preambles)} preambles"
