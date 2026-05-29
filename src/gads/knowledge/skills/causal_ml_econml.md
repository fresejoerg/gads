---
id: causal_ml_econml
description: "EconML/CausalML CATE estimation: CausalForestDML, S/T/X-learners, DML nuisance models with cross-fitting, SHAP on CATE, uplift curves."
triggers: ["CATE", "heterogeneous treatment", "uplift", "double machine learning", "DML", "causal forest", "econml", "causalml", "metalearner", "S-learner", "T-learner", "X-learner", "conditional average treatment effect", "who benefits", "subgroup effect"]
---
# Causal ML: Heterogeneous Treatment Effects (EconML / CausalML)

## CausalForestDML (recommended — EconML)

```python
import numpy as np
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Inputs
# Y: outcome array (n,)
# T: treatment array (n,) — binary 0/1 or continuous
# X: effect modifiers (n, p) — variables that MAY change effect size
# W: baseline controls (n, q) — affect outcome but not effect heterogeneity

est = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=100, random_state=42),
    model_t=RandomForestClassifier(n_estimators=100, random_state=42),
    n_estimators=200,
    random_state=42,
    cv=5        # cross-fitting folds — mandatory
)
est.fit(Y, T, X=X, W=W)

cate_estimates = est.effect(X)          # shape (n,) — per-unit CATE
ate = float(est.ate(X))                 # overall ATE
print(f"ATE = {ate:.4f}")
print(f"CATE range: [{cate_estimates.min():.4f}, {cate_estimates.max():.4f}]")
```

## S / T / X Metalearners (when CausalForest is too heavy)

```python
from econml.metalearners import SLearner, TLearner, XLearner
from sklearn.ensemble import HistGradientBoostingRegressor

# T-Learner: fit separate outcome models per treatment arm
tl = TLearner(models=HistGradientBoostingRegressor(random_state=42))
tl.fit(Y, T, X=np.hstack([X, W]) if W is not None else X)
cate_estimates = tl.effect(X)
ate = float(cate_estimates.mean())
```

## CausalML Uplift (Uber — binary treatment)

```python
from causalml.inference.tree import UpliftRandomForestClassifier

uplift_model = UpliftRandomForestClassifier(
    n_estimators=100,
    evaluationFunction='KL',
    random_state=42
)
uplift_model.fit(X_train, treatment=T_train.astype(str), y=Y_train)
uplift_scores = uplift_model.predict(X_test)
```

## SHAP on CATE Drivers

```python
import shap

# Use TreeExplainer on the underlying forest
shap_values = est.shap_values(X)
# shap_values is a dict; get the 'effects' key
effects_shap = shap_values.get("effects", shap_values)
shap.summary_plot(effects_shap, X, feature_names=feature_names, show=False)
import matplotlib.pyplot as plt
plt.tight_layout()
plt.savefig("cate_shap.png", bbox_inches="tight")
plt.close()
```

## Segmentation by Effect Size

```python
import pandas as pd
df_results = pd.DataFrame({"cate": cate_estimates})
df_results["effect_group"] = pd.qcut(
    df_results["cate"], q=3, labels=["low", "medium", "high"]
)
print(df_results["effect_group"].value_counts())
```

## Sandbox Rules
- `econml`, `causalml`, `xgboost`, `lightgbm` are available (libgomp1 is installed).
- `shap` is available for CATE explanation.
- Use `joblib` for model serialization — pickle is blocked.
- Cross-fitting (`cv=5`) is mandatory; never fit nuisance models on the full dataset without cross-fitting.
