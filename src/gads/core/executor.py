import asyncio
import ast
import uuid
import time
import contextlib
import re
import json
from typing import Optional, List, Tuple, Dict, Any
from gads.agents.workers.coder import CodeGeneratorAgent, CoderInput
from gads.tools.sandbox import SandboxClient, ExecutionResult
from gads.core.models import Artifact, Task
from gads.core.database import engine
from gads.core.bus import bus
from gads.core.runtime_oracle import RuntimeOracle
from gads.core.handover import HandoverManager
from gads.core.error_ledger import (
    normalize_error_reason, record_error, record_resolution, common_pitfalls,
)
from gads.core.llm import CodeGenerationError
from sqlmodel import Session

def _sanitize_code(code: str) -> str:
    """Rewrite broken library imports to sandbox-compatible equivalents.

    lightgbm and xgboost fail at import time (missing libgomp.so.1).
    pickle is blocked by sandbox security policy.
    Replacements happen regardless of what the LLM generated.
    """
    # Strip markdown code fences that leak into the code field. Local models
    # habitually wrap their output in ``` blocks despite the JSON schema; a
    # trailing fence is a guaranteed SyntaxError that burns every retry
    # (observed: gemma-4-12b fenced all 3 generations of a task, each dying
    # at the same parse error before its logic was ever executed).
    code = re.sub(r'^\s*```[a-zA-Z]*\s*$', '', code, flags=re.MULTILINE)

    # lightgbm → sklearn HistGradientBoosting
    code = re.sub(r'from lightgbm import LGBMClassifier', 'from sklearn.ensemble import HistGradientBoostingClassifier', code)
    code = re.sub(r'from lightgbm import LGBMRegressor', 'from sklearn.ensemble import HistGradientBoostingRegressor', code)
    code = re.sub(r'import lightgbm(?:\s+as\s+\w+)?', 'from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor', code)
    code = re.sub(r'lgb\.LGBMClassifier\b', 'HistGradientBoostingClassifier', code)
    code = re.sub(r'lgb\.LGBMRegressor\b', 'HistGradientBoostingRegressor', code)
    code = re.sub(r'\bLGBMClassifier\b', 'HistGradientBoostingClassifier', code)
    code = re.sub(r'\bLGBMRegressor\b', 'HistGradientBoostingRegressor', code)

    # xgboost → sklearn HistGradientBoosting
    code = re.sub(r'from xgboost import XGBClassifier', 'from sklearn.ensemble import HistGradientBoostingClassifier', code)
    code = re.sub(r'from xgboost import XGBRegressor', 'from sklearn.ensemble import HistGradientBoostingRegressor', code)
    code = re.sub(r'import xgboost(?:\s+as\s+\w+)?', 'from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor', code)
    code = re.sub(r'xgb\.XGBClassifier\b', 'HistGradientBoostingClassifier', code)
    code = re.sub(r'xgb\.XGBRegressor\b', 'HistGradientBoostingRegressor', code)
    code = re.sub(r'\bXGBClassifier\b', 'HistGradientBoostingClassifier', code)
    code = re.sub(r'\bXGBRegressor\b', 'HistGradientBoostingRegressor', code)

    # HistGradientBoosting uses max_iter, not n_estimators
    code = re.sub(
        r'(HistGradientBoosting(?:Classifier|Regressor)\s*\([^)]*?)n_estimators\s*=',
        r'\1max_iter=', code
    )

    # pickle → joblib
    code = re.sub(r'\bimport pickle\b', 'import joblib', code)
    code = re.sub(r'\bpickle\.dump\b', 'joblib.dump', code)
    code = re.sub(r'\bpickle\.load\b', 'joblib.load', code)

    # imblearn/imbalanced-learn not installed — strip imports and replace
    # SMOTE/fit_resample with a stratified subsample that actually works.
    if re.search(r'from imblearn|import imblearn', code):
        # Remove import lines
        code = re.sub(r'from imblearn[^\n]*\n', '', code)
        code = re.sub(r'import imblearn[^\n]*\n', '', code)
        # Replace SMOTE/resampler instantiation + fit_resample call patterns:
        # Pattern: X_res, y_res = SMOTE().fit_resample(X, y)
        # → stratified subsample kept at a 5:1 majority:minority ratio
        _smote_replacement = (
            "# imblearn not available — stratified subsample instead\n"
            "_sm_minority_mask = (_y_sm := _sm_y).astype(bool) if hasattr(_sm_y := y, 'values') else y.astype(bool)\n"
        )
        # Simpler: replace fit_resample calls with pass-through (data unchanged)
        code = re.sub(
            r'(\w+)\s*=\s*(?:SMOTE|RandomOverSampler|RandomUnderSampler|BorderlineSMOTE|ADASYN|NearMiss)\s*\([^)]*\)',
            r'# \1 = SMOTE() removed — imblearn not installed',
            code
        )
        code = re.sub(
            r'(\w+)\s*,\s*(\w+)\s*=\s*(\w+)\.fit_resample\((\w+)\s*,\s*(\w+)\)',
            r'\1, \2 = \4, \5  # fit_resample removed — using original data',
            code
        )
        print("  [Sanitizer] Stripped imblearn/SMOTE (not installed in sandbox)", flush=True)

    # HistGradientBoosting auto-numeric patch: fit/predict/predict_proba silently drop
    # non-numeric (string/object) columns from DataFrames. Prevents the common failure
    # where model_a/model_b/prompt/response columns are left in X after a naive .drop().
    # Column selection from fit() is remembered and reused in predict() calls.
    if 'HistGradientBoosting' in code:
        _num_patch = """\
import pandas as _pd_hgb_num
from sklearn.ensemble import HistGradientBoostingClassifier as _HGBCOrig, HistGradientBoostingRegressor as _HGBROrig
_hgb_num_cols: dict = {}
_orig_hgbc_fit = _HGBCOrig.fit
_orig_hgbc_predict = _HGBCOrig.predict
_orig_hgbc_predict_proba = _HGBCOrig.predict_proba
_orig_hgbr_fit = _HGBROrig.fit
_orig_hgbr_predict = _HGBROrig.predict
def _hgb_fit_safe(self, X, y, sample_weight=None):
    if isinstance(X, _pd_hgb_num.DataFrame):
        _hgb_num_cols[id(self)] = X.select_dtypes(include='number').columns.tolist()
        X = X[_hgb_num_cols[id(self)]]
    return _orig_hgbc_fit(self, X, y, sample_weight=sample_weight)
def _hgb_predict_safe(self, X):
    if id(self) in _hgb_num_cols and isinstance(X, _pd_hgb_num.DataFrame):
        X = X[_hgb_num_cols[id(self)]]
    return _orig_hgbc_predict(self, X)
def _hgb_predict_proba_safe(self, X):
    if id(self) in _hgb_num_cols and isinstance(X, _pd_hgb_num.DataFrame):
        X = X[_hgb_num_cols[id(self)]]
    return _orig_hgbc_predict_proba(self, X)
def _hgbr_fit_safe(self, X, y, sample_weight=None):
    if isinstance(X, _pd_hgb_num.DataFrame):
        _hgb_num_cols[id(self)] = X.select_dtypes(include='number').columns.tolist()
        X = X[_hgb_num_cols[id(self)]]
    return _orig_hgbr_fit(self, X, y, sample_weight=sample_weight)
def _hgbr_predict_safe(self, X):
    if id(self) in _hgb_num_cols and isinstance(X, _pd_hgb_num.DataFrame):
        X = X[_hgb_num_cols[id(self)]]
    return _orig_hgbr_predict(self, X)
_HGBCOrig.fit = _hgb_fit_safe
_HGBCOrig.predict = _hgb_predict_safe
_HGBCOrig.predict_proba = _hgb_predict_proba_safe
_HGBROrig.fit = _hgbr_fit_safe
_HGBROrig.predict = _hgbr_predict_safe
"""
        code = _num_patch + "\n" + code

    # HistGradientBoosting has no native .feature_importances_ — patch it via split-gain sums.
    # Triggered when HGB is in code OR when feature_importances_ is accessed (the model may have
    # been created by a previous task and exist only in the kernel, not in this code block).
    if 'feature_importances_' in code:
        _fi_patch = """\
import numpy as _np_hgb_fi
def _hgb_fi_property(self):
    _fi = _np_hgb_fi.zeros(self.n_features_in_)
    for _stage in self._predictors:
        for _predictor in _stage:
            for _node in _predictor.nodes:
                if not bool(_node['is_leaf']):
                    _fi[int(_node['feature_idx'])] += max(0.0, float(_node['gain']))
    _s = _fi.sum()
    return _fi / _s if _s > 0 else _fi
from sklearn.ensemble import HistGradientBoostingClassifier as _HGBCls, HistGradientBoostingRegressor as _HGBReg
if not isinstance(vars(_HGBCls).get('feature_importances_'), property):
    _HGBCls.feature_importances_ = property(_hgb_fi_property)
    _HGBReg.feature_importances_ = property(_hgb_fi_property)
"""
        code = _fi_patch + "\n" + code

    # Multiclass predict_proba binary-slice fix:
    # Replace predict_proba(...)[:, 1] with predict_proba(...) so log_loss gets full matrix.
    # Only applied when log_loss is also used (classification context).
    if 'log_loss' in code and re.search(r'\.predict_proba\([^)]+\)\[:,\s*1\]', code):
        code = re.sub(
            r'(\.predict_proba\([^)]+\))\[:,\s*1\]',
            r'\1',
            code
        )
        print("  [Sanitizer] Removed binary predict_proba slice (multiclass context detected)", flush=True)

    # PyMC/Bambi multiprocessing fix: chains>1 forks processes in Docker which hang
    # indefinitely after sampling. Force single-chain execution regardless of LLM output.
    if re.search(r'\bchains\s*=\s*[2-9]\d*', code):
        code = re.sub(r'\bchains\s*=\s*[2-9]\d*', 'chains=1', code)
        print("  [Sanitizer] Forced chains=1 (multiprocessing unsafe in Docker sandbox)", flush=True)
    if re.search(r'\bcores\s*=\s*[2-9]\d*', code):
        code = re.sub(r'\bcores\s*=\s*[2-9]\d*', 'cores=1', code)
        print("  [Sanitizer] Forced cores=1 (multiprocessing unsafe in Docker sandbox)", flush=True)

    # Unregularized LR performance fix: penalty=None on large datasets diverges slowly.
    # Replace with C=1.0 (L2) for stable, fast convergence.
    if re.search(r'LogisticRegression\s*\([^)]*penalty\s*=\s*None', code):
        code = re.sub(r'\bpenalty\s*=\s*None', 'C=1.0', code)
        print("  [Sanitizer] Replaced penalty=None with C=1.0 in LogisticRegression (performance)", flush=True)

    # General ML training large-dataset guard: if code loads a CSV/parquet AND trains a
    # classifier or regressor but has no .sample() call, inject a 50K row cap.
    # AutoGluon, sklearn, xgboost, lightgbm all benefit; prevents Coder generation timeouts
    # on 100K+ row datasets when the Planner forgot the sample_rows constraint.
    _TRAINING_MARKERS = ['TabularPredictor', 'fit(', 'XGBClassifier', 'LGBMClassifier',
                         'RandomForestClassifier', 'GradientBoosting', 'HistGradientBoosting']
    _HAS_FILE_LOAD = re.search(r'read_csv|read_parquet|read_excel', code)
    _HAS_TRAINING = any(m in code for m in _TRAINING_MARKERS)
    _HAS_SAMPLE = 'sample(' in code or '.sample(' in code
    # Collaborative filtering must NEVER get the random row cap: interaction logs are
    # long-tailed, so a random subset shares almost no users/items and the k-core filter
    # downstream collapses the matrix (an 800k-review log became 10x13). Those recipes do
    # their own density-preserving reduction via gads_build_interaction_matrix(max_rows=...).
    # Note `_TRAINING_MARKERS` contains 'fit(', which CF's model.fit(matrix) matches.
    _CF_MARKERS = ('gads_build_interaction_matrix', 'gads_dense_core_sample',
                   'gads_recommend_and_evaluate', 'AlternatingLeastSquares', 'implicit',
                   'interaction_matrix', 'csr_matrix')
    _IS_CF = any(m in code for m in _CF_MARKERS)
    if _IS_CF and _HAS_FILE_LOAD and _HAS_TRAINING and not _HAS_SAMPLE:
        print("  [Sanitizer] Skipped row-cap guard: collaborative-filtering code "
              "(random sampling would destroy interaction density)", flush=True)
    if _HAS_FILE_LOAD and _HAS_TRAINING and not _HAS_SAMPLE and not _IS_CF:
        _ml_subsample_injection = (
            "# Auto-injected ML training guard: cap dataset at 50K rows to prevent timeout\n"
            "_ml_row_limit = 50_000\n"
            "if 'df' in dir() and hasattr(df, '__len__') and len(df) > _ml_row_limit:\n"
            "    print(f'[SampleGuard] Subsampling from {len(df):,} to {_ml_row_limit:,} rows')\n"
            "    _target = locals().get('target_col', globals().get('target_col', None))\n"
            "    if _target is None and 'target' in locals(): _target = locals()['target']\n"
            "    if _target is None and 'target' in globals(): _target = globals()['target']\n"
            "    _stratify_col = None\n"
            "    if _target is not None and _target in df.columns:\n"
            "        _vc = df[_target].value_counts()\n"
            "        if len(_vc) >= 2 and len(_vc) <= 20:\n"
            "            _stratify_col = _target\n"
            "    try:\n"
            "        if _stratify_col is not None:\n"
            "            _frac = _ml_row_limit / len(df)\n"
            "            df = df.groupby(_stratify_col, group_keys=False).apply(lambda x: x.sample(max(1, int(len(x) * _frac)), random_state=42, replace=False) if len(x) > 0 else x).reset_index(drop=True)\n"
            "            print('[SampleGuard] Stratified sampling applied successfully')\n"
            "        else:\n"
            "            df = df.sample(_ml_row_limit, random_state=42).reset_index(drop=True)\n"
            "    except Exception as e:\n"
            "        print(f'[SampleGuard] Stratified sampling failed: {e}. Falling back to random sampling.')\n"
            "        df = df.sample(_ml_row_limit, random_state=42).reset_index(drop=True)\n"
        )
        # Inject after the last read_csv/read_parquet line
        _load_match = list(re.finditer(r'(df\s*=\s*pd\.read_(?:csv|parquet|excel)\([^\n]+\))', code))
        if _load_match:
            last = _load_match[-1]
            insert_pos = last.end()
            code = code[:insert_pos] + "\n" + _ml_subsample_injection + code[insert_pos:]
        else:
            code = _ml_subsample_injection + code
        print("  [Sanitizer] Injected 50K ML training subsample guard", flush=True)

    # Remove hallucinated AutoGluon imports (these packages don't exist)
    if 'AutogluonModels' in code or 'from gads_emit_insight import' in code or 'autogluon.models' in code:
        code = re.sub(r'from AutogluonModels import [^\n]+\n?', '', code)
        code = re.sub(r'from gads_emit_insight import [^\n]+\n?', '', code)
        code = re.sub(r'from autogluon\.models import [^\n]+\n?', '', code)
        code = re.sub(r'import autogluon\.models[^\n]*\n?', '', code)
        print("  [Sanitizer] Removed hallucinated AutogluonModels/autogluon.models/gads_emit_insight imports", flush=True)

    # Remove hallucinated imports of the gads native helpers. Functions like
    # gads_causal_estimate_ate, gads_emit_insight, gads_automl_fit, and
    # gads_calibrate_threshold are INJECTED into the kernel by the executor preamble —
    # they are not importable modules. Local models often invent a wrapper module
    # (e.g. `import gads_utils`, `from gads_helpers import gads_causal_estimate_ate`),
    # which raises ModuleNotFoundError and, in local_only mode, cannot escalate away.
    # Stripping the import line is safe: there is no legitimate `gads*` package to import.
    if re.search(r'(?m)^[ \t]*(?:from|import)[ \t]+gads', code):
        code = re.sub(r'(?m)^[ \t]*from[ \t]+gads\w*[ \t]+import[^\n]*\n?', '', code)
        code = re.sub(r'(?m)^[ \t]*import[ \t]+gads\w*(?:[ \t]+as[ \t]+\w+)?[ \t]*\n?', '', code)
        print("  [Sanitizer] Removed hallucinated gads_* module imports (native helpers are kernel-injected)", flush=True)

    # Remove hallucinated causal library imports that do not exist as installed packages.
    # Note: `causalinference` (pip: CausalInference) IS a real package — do NOT strip it.
    _causal_hallucinations = ['causal_models', 'causal_inference_lib']
    if any(h in code for h in _causal_hallucinations):
        code = re.sub(r'(?:from causal_models|import causal_models)[^\n]*\n?', '', code)
        code = re.sub(r'(?:from causal_inference_lib|import causal_inference_lib)[^\n]*\n?', '', code)
        print("  [Sanitizer] Removed hallucinated causal library imports (causal_models/causal_inference_lib)", flush=True)

    # Strip causal_learn imports when used for DoWhy estimation tasks (wrong library).
    # The package is pip install causal-learn; correct import name is `causallearn` (no underscore).
    # Models sometimes write `from causal_learn import ...` (wrong name) or use it for effect
    # estimation (wrong purpose — causallearn is for structure discovery: PC/FCI/GES).
    # Strip the bad import; the skill/recipe provides the correct dowhy API instead.
    if 'causal_learn' in code or 'from cdt' in code:
        code = re.sub(r'(?:from causal_learn|import causal_learn)[^\n]*\n?', '', code)
        code = re.sub(r'(?:from cdt|import cdt)[^\n]*\n?', '', code)
        print("  [Sanitizer] Removed causal_learn/cdt imports (wrong library for DoWhy estimation)", flush=True)

    # Strip mock CausalModel class definitions — local model sometimes fakes the DoWhy class.
    # Pattern: class CausalModel: or class CausalModel(object): followed by indented body.
    # Removing the entire class definition prevents the fake object from shadowing the real one.
    if re.search(r'^class CausalModel[\s:(]', code, re.MULTILINE):
        code = re.sub(r'^class CausalModel[\s\S]*?(?=\n\S|\Z)', '', code, flags=re.MULTILINE)
        print("  [Sanitizer] Removed mock CausalModel class definition", flush=True)

    # Strip invalid backslash line-continuations inside parentheses and \) escape sequences.
    # Local model sometimes writes: func(arg1, \   or  method\) — both are syntax errors.
    # Safe to remove: inside open parens Python does implicit line continuation; \) is never valid.
    if '\\)' in code or re.search(r'\\ *\n', code):
        code = re.sub(r'\\\)', ')', code)               # \) → )
        code = re.sub(r'\\ *\n', '\n', code)            # trailing \ at end of line → bare newline
        print("  [Sanitizer] Stripped invalid backslash continuations (\\) and trailing \\)", flush=True)

    # Replace literal \n (backslash-n as two chars) before closing delimiters with a real newline.
    # Local model sometimes writes: CausalModel(graph=gml_string,\n) — syntax error outside strings.
    if '\\n' in code and re.search(r'\\n[)\],}]', code):
        code = re.sub(r'\\n([)\],}])', r'\n\1', code)
        print("  [Sanitizer] Replaced literal \\n-before-delimiter with real newline", flush=True)

    # Fix gml_string split across lines: model breaks a string literal at \n by inserting an actual newline.
    # Pattern: a gml_string assignment line ends with + " (opening an unclosed string literal),
    # and the next line contains the closing ".  Join them back into one line.
    if 'gml_string' in code:
        lines = code.split('\n')
        fixed_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if 'gml_string' in line:
                stripped = line.rstrip()
                if stripped.endswith('+ "') or stripped.endswith("+ '"):
                    if i + 1 < len(lines):
                        fixed_lines.append(stripped + lines[i + 1].lstrip())
                        i += 2
                        print("  [Sanitizer] Joined split gml_string concatenation across lines", flush=True)
                        continue
            fixed_lines.append(line)
            i += 1
        code = '\n'.join(fixed_lines)

    # Unwrap def main(): — variables inside a function are local and invisible to other tasks
    if re.search(r'^def main\(\):', code, re.MULTILINE):
        lines = code.split('\n')
        new_lines, in_body, skip_rest = [], False, False
        for line in lines:
            if skip_rest:
                continue
            if line.rstrip() == 'def main():':
                in_body = True
                continue
            if in_body:
                if line.startswith('    '):
                    stripped = line[4:]
                    if stripped.strip() in ('return None', 'return'):
                        continue
                    new_lines.append(stripped)
                elif line.strip() == '':
                    if not new_lines or new_lines[-1] != '':
                        new_lines.append('')
                else:
                    in_body = False
                    if line.strip().startswith('if __name__'):
                        skip_rest = True
                    else:
                        new_lines.append(line)
            else:
                new_lines.append(line)
        code = '\n'.join(new_lines)
        print("  [Sanitizer] Unwrapped def main(): → global scope", flush=True)

    # Fix capitalized CSV filename: Creditcard.csv / CreditCard.csv → creditcard.csv
    code = re.sub(r"pd\.read_csv\(['\"]([Cc]redit[Cc]ard\.csv)['\"]\)", "pd.read_csv('creditcard.csv')", code)

    # Ensure df_clean is defined before use — guard against model defining it inside a branch
    if 'df_clean' in code and 'df_clean = ' not in code and 'df_clean=' not in code:
        code = "df_clean = df.drop(columns=drop_cols if 'drop_cols' in dir() else [], errors='ignore')\n" + code
        print("  [Sanitizer] Injected df_clean = df.drop(...) guard at top", flush=True)

    # Fix AutoGluon .fit() argument: Presets= → presets= (case-sensitive keyword)
    code = re.sub(r"\bPresets\s*=", "presets=", code)
    if 'presets=' in code.lower() and 'Presets=' not in code:
        pass  # already fixed or was correct
    elif re.search(r'\bPresets\s*=', code):
        print("  [Sanitizer] Fixed Presets= → presets= in .fit() call", flush=True)

    # Fix "unexpected character after line continuation" — trailing space after backslash
    if '\\ \n' in code or '\\\t\n' in code:
        code = re.sub(r'\\[ \t]+\n', '\n', code)
        print("  [Sanitizer] Removed trailing space after backslash line continuation", flush=True)

    # Fix stratify=target_col (string) → stratify=df[target_col] (Series) in train_test_split
    if 'stratify=target_col' in code:
        code = code.replace('stratify=target_col', 'stratify=df_clean[target_col] if "df_clean" in dir() else df[target_col]')
        print("  [Sanitizer] Fixed stratify=target_col → stratify=df[target_col]", flush=True)

    # Fix common local-model typo: target_mol (OCR-like confusion) → target_col
    if 'target_mol' in code:
        code = code.replace('target_mol', 'target_col')
        print("  [Sanitizer] Fixed target_mol → target_col typo", flush=True)

    # Fix common local-model typo: minority_class_fract → minority_class_frac
    if 'minority_class_fract' in code:
        code = code.replace('minority_class_fract', 'minority_class_frac')
        print("  [Sanitizer] Fixed minority_class_fract → minority_class_frac typo", flush=True)

    # Fix duplicate closing square brackets on df access: df[col]] → df[col]
    if 'df[' in code:
        code = re.sub(r'df\[([^\]\n]+)\]\]', r'df[\1]', code)
        print("  [Sanitizer] Fixed duplicate closing brackets on df", flush=True)

    # Fix dictionary keys from gads_causal_estimate_ate return
    if 'gads_causal_estimate_ate' in code:
        code = code.replace('result["subset_new_effectiveness"]', 'result["subset_new_effect"]')
        code = code.replace('result["placebo_new_effectiveness"]', 'result["placebo_new_effect"]')
        code = code.replace('result["causal_"]', 'result["causal_estimate"]')
        code = code.replace('result["causal_estimate_"]', 'result["causal_estimate"]')
        code = code.replace('treatment=treatment_lane', 'treatment=treatment_col')
        print("  [Sanitizer] Standardized gads_causal_estimate_ate return keys", flush=True)

    # AutoGluon predict_proba returns a DataFrame, not ndarray; fix numpy-style [:, n] indexing
    if 'predict_proba' in code or ('y_prob' in code and '[:,' in code):
        code = re.sub(r'predict_proba\(([^)]*)\)\[:,\s*(\d+)\]', r'predict_proba(\1).iloc[:, \2]', code)
        code = re.sub(r'\by_prob\[:,\s*(\d+)\]', r'y_prob.iloc[:, \1]', code)
        if 'predict_proba' in code:
            print("  [Sanitizer] Fixed predict_proba numpy-style indexing → .iloc", flush=True)

    # Fix non-existent pandas API: is_datetime64_ns → is_datetime64_any_dtype
    if 'is_datetime64_ns' in code:
        code = code.replace('pd.api.types.is_datetime64_ns(', 'pd.api.types.is_datetime64_any_dtype(')
        code = code.replace('.is_datetime64_ns(', '.is_datetime64_any_dtype(')
        print("  [Sanitizer] Replaced is_datetime64_ns with is_datetime64_any_dtype", flush=True)

    # AutoGluon feature_importance timeout guard: permutation importance over the full
    # test set (10K+ rows, 5 shuffle sets) easily exceeds the 600s sandbox limit.
    # Force subsample_size=1000, num_shuffle_sets=1 only when the call is completely
    # unguarded. A num_shuffle_sets kwarg means the author already addressed the cost
    # (the skill pattern pre-subsamples into fi_df) — appending here would duplicate
    # the kwarg and turn valid code into a SyntaxError.
    if 'feature_importance(' in code and 'subsample_size' not in code and 'num_shuffle_sets' not in code:
        code = re.sub(
            r'\.feature_importance\(([^)]*?)\)',
            lambda m: '.feature_importance(' + m.group(1).rstrip(', ') +
                      (', ' if m.group(1).strip() else '') +
                      'subsample_size=1000, num_shuffle_sets=1)',
            code
        )
        print("  [Sanitizer] Injected subsample_size=1000, num_shuffle_sets=1 into feature_importance() call", flush=True)

    # DoWhy CausalModel large-dataset guard: inject a 20K subsample before CausalModel
    # construction if no subsample is present. PSM/DML on 284K rows times out.
    if 'CausalModel' in code and 'sample(' not in code and 'df_sample' not in code:
        _subsample_injection = (
            "# Auto-injected subsample guard: causal estimation needs ≤20K rows\n"
            "_causal_n_limit = 20000\n"
            "if len(df) > _causal_n_limit:\n"
            "    print(f'Subsampling df from {len(df)} to {_causal_n_limit} rows for causal estimation')\n"
            "    df = df.sample(_causal_n_limit, random_state=42).reset_index(drop=True)\n"
        )
        code = _subsample_injection + code
        print("  [Sanitizer] Injected 20K subsample guard for CausalModel usage", flush=True)

    code = _repair_native_kwarg_case(code)
    code = _repair_stray_indent(code)
    return code


def _repair_native_kwarg_case(code: str) -> str:
    """Lower-case capitalized keyword arguments in calls to gads_* native nodes.

    Small local models capitalize identifiers unpredictably — observed across runs:
    `user_col` -> `User_Col`, `method` -> `Method`. Against a native with a fixed signature
    that is an immediate `TypeError: unexpected keyword argument`, and each variant is a
    *different* error string, so the adaptive retry policy never sees a repeat and burns the
    whole budget on a one-character problem.

    Safe because it is signature-verified: a name is rewritten only when the lower-cased form
    is a real parameter of that specific native AND the written form is not. Unknown functions
    and genuinely wrong kwargs are left alone for the normal error feedback to handle.
    """
    import re as _re
    if "gads_" not in code:
        return code
    try:
        import inspect as _inspect
        from gads.knowledge.native import NATIVE_REGISTRY
        params = {name: set(_inspect.signature(fn).parameters)
                  for name, fn in NATIVE_REGISTRY.items()}
    except Exception:
        return code

    fixed = []

    def _fix_call(m):
        fname, args = m.group(1), m.group(2)
        valid = params.get(fname)
        if not valid:
            return m.group(0)

        def _fix_kw(km):
            kw = km.group(1)
            if kw in valid:
                return km.group(0)
            low = kw.lower()
            if low in valid:
                fixed.append(f"{fname}({kw}->{low})")
                return f"{low}{km.group(2)}"
            return km.group(0)

        # Only rewrite top-level `Name =` kwarg positions, never `==` comparisons.
        new_args = _re.sub(r"\b([A-Za-z_]\w*)(\s*=(?!=))", _fix_kw, args)
        return f"{fname}({new_args}"

    # Match up to the call's argument text; nested parens are handled by the kwarg regex
    # operating on whatever it captured, which is sufficient for these flat native calls.
    out = _re.sub(r"\b(gads_\w+)\(([^)]*)", _fix_call, code)
    if fixed:
        print(f"  [Sanitizer] Fixed native kwarg casing: {', '.join(sorted(set(fixed))[:5])}",
              flush=True)
    return out


def _repair_stray_indent(code: str) -> str:
    """Strip stray leading whitespace that makes otherwise-valid code unparseable.

    Small local models routinely emit a top-level statement with one accidental leading
    space — "unexpected indent" is one of the most common failure signatures we see, and it
    burns the whole retry budget on code that is otherwise correct (an observed case differed
    from working code by exactly one character, on an optional insight-emitting call).

    Only fixes lines that are indented while the preceding logical line is itself at column 0
    and does NOT open a block, and only accepts the result if it actually parses. Verification
    by `ast.parse` is what makes this safe: a repair that changes meaning cannot be accepted,
    and anything unparseable for another reason is returned untouched for the normal
    error-feedback loop to handle.
    """
    import ast as _ast
    try:
        _ast.parse(code)
        return code
    except SyntaxError:
        pass

    lines = code.split("\n")
    prev_idx = None
    repaired, fixed = list(lines), 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent > 0 and prev_idx is not None:
            # Read the REPAIRED previous line, not the original: when a model emits a
            # whole program uniformly indented (every line after the first at col 4),
            # only line 2 has an original predecessor at col 0. Comparing against the
            # original stopped the repair dead after one line, the partial result
            # failed to parse, and the entire repair was discarded. Cascading down the
            # repaired text flattens the whole block; `opens_block` still guards real
            # nesting, since a dedented `def foo():` correctly blocks its body.
            prev = repaired[prev_idx]
            prev_s = prev.strip()
            prev_indent = len(prev) - len(prev.lstrip())
            opens_block = prev_s.endswith((":", "\\", ",", "(", "[", "{"))
            if prev_indent == 0 and not opens_block:
                repaired[i] = line.lstrip()
                fixed += 1
        prev_idx = i

    if fixed:
        candidate = "\n".join(repaired)
        try:
            _ast.parse(candidate)
            print(f"  [Sanitizer] Repaired {fixed} stray-indent line(s) "
                  f"(verified by re-parsing)", flush=True)
            return candidate
        except SyntaxError:
            pass
    return code


def _truncate_error_msg(msg, limit=1200):
    """Cap one attempt's error text before it is fed back to the Coder.

    A single sklearn error can carry multi-KB of concatenated label values
    ("Labels in y_true and y_pred ... Got y_true=[...]"), which crowds out the task
    and the prior attempts on a small local context window. Keep the head (exception
    type + the start of the message, where the actual cause is) and the tail (where
    assertion detail usually sits), drop the middle.
    """
    msg = str(msg)
    if len(msg) <= limit:
        return msg
    head = int(limit * 0.7)
    tail = limit - head
    return (msg[:head] + f"\n    … [{len(msg) - limit} chars of error text elided] …\n"
            + msg[-tail:])


class ExecutionManager:
    """Manages the Code-Execution-Feedback loop."""

    def __init__(self, sandbox_url: str = "http://localhost:8000"):
        self.coder = CodeGeneratorAgent()
        self.sandbox = SandboxClient(base_url=sandbox_url)
        self.handover = HandoverManager(sandbox_url=sandbox_url)
        self.authoritative_state: Dict[str, Any] = {}
        self.file_schemas: Dict[str, Any] = {}  # populated by server after schema probe

    async def run_task(
        self, 
        task_description: str, 
        project_id: uuid.UUID, 
        session_id: str = "default",
        skills_context: Optional[str] = None,
        task_id: Optional[uuid.UUID] = None,
        stdout_callback = None,
        stream_callback = None,
        cancel_check = None,
        state_summary: Optional[str] = None,
        recipe_id: Optional[str] = None,
        recipe_version: Optional[str] = None,
        fallback_native: Optional[str] = None,
        fallback_call: Optional[str] = None,
        fallback_mode: str = "none",
        max_attempts: int = 10,
    ) -> Tuple[ExecutionResult, str]:
        """
        Runs the full loop with State Introspection. 
        Does NOT handle DB persistence (caller must do that).
        """
        # Adaptive retry policy: keep retrying (up to `max_attempts`) as long as each
        # failure is for a DIFFERENT reason — evidence the model is self-correcting on the
        # accumulated error feedback. Stop as soon as the SAME reason recurs: a repeated
        # error means the model is stuck, not progressing, so further re-rolls only waste
        # time. ALL prior errors are fed back each attempt (the error text alone, not the
        # code — the error is the signal that nudges a different approach).
        # `max_attempts` is a parameter: the cloud fallback re-invokes run_task with a cloud
        # model and max_attempts=1 (a single escalated attempt, not another full retry loop).
        retry_count = 0
        error_feedback = None
        previous_code = ""
        error_history = []           # one human-readable error string per failed attempt
        error_reason_counts = {}     # normalized reason -> count (a reason seen twice => stop)
        last_bad_text = ""           # last unparseable generation, for the failure payload

        # First-attempt prior: recurring structural failures this recipe step has hit in
        # PAST runs (from the cross-run error ledger), injected so the model can preempt
        # known dead ends before it makes them. None if off-recipe or nothing recurs yet.
        pitfalls_prior = common_pitfalls(recipe_id, task_description)
        if pitfalls_prior:
            print(f"    [Executor] Injecting {pitfalls_prior.count(chr(10)) + 1} known "
                  f"pitfall(s) for this recipe step (from error ledger)", flush=True)

        # Buffer for reasoning debouncing
        reasoning_buffer = []
        last_emit = 0.0

        async def debounced_stream_callback(token: str):
            nonlocal last_emit
            if not stream_callback: return
            reasoning_buffer.append(token)
            now = time.time()
            if now - last_emit > 0.15:
                delta = "".join(reasoning_buffer)
                reasoning_buffer.clear()
                last_emit = now
                await stream_callback(delta)

        async def poll_logs_loop():
            if not stdout_callback: return
            offset_out = 0
            offset_err = 0
            accumulated_out = ""
            accumulated_err = ""
            print(f"    [Executor] Starting log poller for session {session_id}", flush=True)
            while True:
                await asyncio.sleep(1.0)
                logs = await self.sandbox.poll_logs(session_id, offset_out, offset_err)
                if logs:
                    new_out = logs.get("stdout", "")
                    new_err = logs.get("stderr", "")
                    offset_out = logs.get("offset_out", offset_out)
                    offset_err = logs.get("offset_err", offset_err)
                    
                    if new_out or new_err:
                        print(f"    [Executor] Polled {len(new_out)} chars of new stdout from {session_id}", flush=True)
                        accumulated_out += new_out
                        accumulated_err += new_err
                        combined = accumulated_out + "\n" + accumulated_err
                        
                        # Collapse carriage returns for tqdm
                        lines = combined.split('\n')
                        cleaned_lines = []
                        for line in lines:
                            if '\r' in line:
                                line = line.split('\r')[-1]
                            cleaned_lines.append(line)
                        cleaned_text = '\n'.join(cleaned_lines)
                        
                        await stdout_callback(cleaned_text)

        while retry_count < max_attempts:
            print(f"    [Executor] --- Step {retry_count + 1} for: {task_description[:30]}... ---", flush=True)
            
            try:
                # 0. Early Exit if Cancelled
                if cancel_check and await cancel_check():
                    print(f"    [Executor] 🛑 Aborting task due to user cancellation.", flush=True)
                    return ExecutionResult(
                        stdout="", stderr="Workflow cancelled by user.", 
                        error={"ename": "Cancelled", "evalue": "User requested abort"},
                        execution_time_ms=0,
                        kernel_state={}
                    ), self.coder.model

                available_files = self.sandbox.list_workspace_files(project_id)
                
                print(f"    [Executor] Calling {self.coder.model} with {len(self.authoritative_state)} variables...", flush=True)
                
                # Fetch postcondition contract for the worker
                contract = None
                task_escalations = 0
                with Session(engine) as session:
                    from gads.core.models import Task as DBTask
                    t_obj = session.get(DBTask, task_id)
                    if t_obj:
                        contract = t_obj.postcondition_json
                        task_escalations = t_obj.escalation_count or 0

                # Label the upcoming Coder generation with its attempt number so
                # first-shot and retried completions are distinguishable in Langfuse
                # (telemetry plan 010, Phase 1b).
                from gads.core.llm import trace_context
                ctx = trace_context.get()
                if ctx is not None:
                    ctx.update({"attempt": retry_count + 1, "escalation_count": task_escalations})

                coder_res = await asyncio.wait_for(
                    self.coder.run(CoderInput(
                        task_description=task_description,
                        available_files=available_files,
                        file_schemas=self.file_schemas,
                        authoritative_state=self.authoritative_state,
                        previous_code=previous_code,
                        error_feedback=error_feedback,
                        common_pitfalls=(pitfalls_prior if retry_count == 0 else None),
                        skills_context=skills_context,
                        task_id=str(task_id) if task_id else None,
                        postcondition_contract=contract,
                        state_summary=state_summary
                    ), stream_callback=debounced_stream_callback),
                    timeout=300.0
                )
                
                # Flush remaining reasoning
                if stream_callback and reasoning_buffer:
                    delta = "".join(reasoning_buffer)
                    reasoning_buffer.clear()
                    await stream_callback(delta)
                
                current_code = _sanitize_code(coder_res.content.code)

                # Honesty gate: everything that reaches the sandbox must be a program.
                # A reasoning model that never closes its fence leaks deliberation prose,
                # which the sandbox reports as "ValidationError: syntax error at line n" —
                # a diagnosis the model cannot act on, burning every retry. Checking here
                # (after the sanitizer has had its repair pass, so legitimately fixable
                # code is not rejected) routes any non-program into the CodeGenerationError
                # branch with an actionable remedy, and saves a sandbox round-trip.
                try:
                    ast.parse(current_code)
                except SyntaxError as _se:
                    # Report the error the model must actually FIX. When a generation has
                    # both a uniform-indent artifact and a real syntax error, the repair
                    # above is discarded (it only commits a repair that parses), so the
                    # surfaced error is the indent — a harness-fixable formatting artifact.
                    # The model then "fixes" indentation, hits the same message, and the
                    # same-reason guard stops it, with the true defect never mentioned.
                    # Re-check against a best-effort dedent: if that yields a DIFFERENT
                    # error, that residual is the honest diagnosis.
                    _msg, _line = _se.msg, _se.lineno
                    _lines = current_code.split("\n")
                    _dedented = "\n".join(
                        [_lines[0]] + [ln[4:] if ln.startswith("    ") else ln
                                       for ln in _lines[1:]])
                    try:
                        ast.parse(_dedented)
                    except SyntaxError as _se2:
                        if _se2.msg != _se.msg:
                            _msg, _line = _se2.msg, _se2.lineno
                    _err = CodeGenerationError(
                        f"generated text is not valid Python ({_msg} at line {_line})",
                        content_chars=len(current_code),
                    )
                    _err.text = current_code
                    raise _err from None

                # --- PREDICTIVE RUNTIME ORACLE ---
                # 1. Gather Data Dimensions
                n_rows, m_cols = 0, 0
                for var_info in self.authoritative_state.values():
                    if var_info.get("type") == "DataFrame":
                        shape = var_info.get("shape", [0, 0])
                        n_rows = max(n_rows, shape[0])
                        m_cols = max(m_cols, shape[1])
                
                # 2. Estimate
                est_seconds = RuntimeOracle.estimate_runtime(current_code, n_rows, m_cols)
                print(f"    [Oracle] Estimated Runtime: {est_seconds:.1f}s (N={n_rows}, M={m_cols})", flush=True)

                if task_id:
                    with Session(engine) as session:
                        from gads.core.models import Task as DBTask
                        t_obj = session.get(DBTask, task_id)
                        if t_obj: 
                            t_obj.estimated_runtime_s = est_seconds
                            session.add(t_obj)
                            session.commit()

                # 3. Decision Branch (300s limit)
                if est_seconds > 280.0: # 20s buffer
                    print(f"    [Executor] ⚠️ TASK BYPASSED: Likely to exceed safety limit. Generating handover bundle...", flush=True)
                    bundle_file = await self.handover.create_bundle(project_id, current_code, est_seconds)
                    
                    return ExecutionResult(
                        stdout=f"BYPASSED: Task estimated to take {est_seconds/60:.1f} minutes. Handover bundle created: {bundle_file}",
                        stderr="",
                        result=f"HANDOVER_BUNDLE:{bundle_file}",
                        execution_time_ms=0,
                        kernel_state={}
                    ), coder_res.model_used

                # 0.5. Check Cancellation before Sandbox execution
                if cancel_check and await cancel_check():
                    print(f"    [Executor] 🛑 Aborting execution due to user cancellation.", flush=True)
                    return ExecutionResult(
                        stdout="", stderr="Workflow cancelled by user.", 
                        error={"ename": "Cancelled", "evalue": "User requested abort"},
                        execution_time_ms=0,
                        kernel_state={},
                        code=current_code
                    ), self.coder.model

                print(f"    [Executor] Executing code in sandbox...", flush=True)

                # Inject the native-node definitions this code references (keyword-routed;
                # single source of truth in gads.knowledge.native.preamble_for_code, shared
                # with kernel rehydration so replayed code gets the same definitions).
                _native_preamble = ""
                try:
                    from gads.knowledge.native import preamble_for_code
                    _native_preamble, _native_names = preamble_for_code(current_code)
                    if _native_names:
                        print(f"    [Executor] Injecting native node preamble(s): "
                              f"{', '.join(_native_names)}", flush=True)
                except Exception as _e:
                    print(f"    [Executor] Warning: Could not load native preambles: {_e}", flush=True)

                # Wrap code with telemetry hooks
                telemetry_preamble = _native_preamble + """
if '_gads_insights' not in globals(): _gads_insights = []
def gads_emit_insight(artifact, insight, evidence=""):
    _gads_insights.append({"artifact": artifact, "insight": insight, "evidence": evidence})
"""
                telemetry_postamble = """
import json as _json
print("GADS_INSIGHTS_JSON:" + _json.dumps(_gads_insights))
_gads_insights = [] # Clear for next task
"""
                wrapped_code = telemetry_preamble + "\n" + current_code + "\n" + telemetry_postamble

                poller = asyncio.create_task(poll_logs_loop())
                try:
                    _sandbox_timeout = 720.0 if "local_model" in (self.coder.model_str or "") else 360.0
                    exec_result = await asyncio.wait_for(
                        self.sandbox.execute(wrapped_code, project_id=project_id, session_id=session_id),
                        timeout=_sandbox_timeout
                    )
                    exec_result.code = current_code # Attach ORIGINAL code for persistence
                    
                    # Parse Semantic Insights
                    if "GADS_INSIGHTS_JSON:" in exec_result.stdout:
                        try:
                            parts = exec_result.stdout.split("GADS_INSIGHTS_JSON:")
                            raw_json = parts[1].strip().split("\n")[0]
                            exec_result.semantic_insights = json.loads(raw_json)
                            exec_result.stdout = parts[0] + "\n".join(parts[1].strip().split("\n")[1:])
                        except Exception as e:
                            print(f"    [Executor] Warning: Failed to parse semantic insights: {e}")

                    # --- AUTOMATIC STRUCTURAL TELEMETRY (The 'Verification Floor') ---
                    # Probe kernel for DataFrames and extract deterministic stats
                    structural_probe_code = """
import json as _json
import pandas as _pd
import numpy as _np
_floor = []
for _name, _obj in {**globals(), **locals()}.items():
    if isinstance(_obj, _pd.DataFrame):
        try:
            _floor.append({
                "artifact": _name,
                "insight": f"Structural Floor: {_name} ({_obj.shape[0]}x{_obj.shape[1]})",
                "evidence": f"Columns: {list(_obj.columns)} | Nulls: {_obj.isna().sum().sum()} | Unique Samples: {_json.dumps({c: str(_obj[c].nunique()) for c in _obj.columns[:5]})}",
                "is_floor": True
            })
        except: pass
print("GADS_FLOOR_JSON:" + _json.dumps(_floor))
"""
                    try:
                        floor_res = await self.sandbox.execute(structural_probe_code, project_id=project_id, session_id=session_id)
                        if "GADS_FLOOR_JSON:" in floor_res.stdout:
                            raw_floor = floor_res.stdout.split("GADS_FLOOR_JSON:")[1].strip().split("\n")[0]
                            floor_data = json.loads(raw_floor)
                            exec_result.semantic_insights.extend(floor_data)
                    except Exception as e:
                        print(f"    [Executor] Warning: Structural probe failed: {e}")

                    # LOG ACTUAL RUNTIME FOR LEARNING
                    # Scan for first estimator to log
                    estimators = RuntimeOracle.analyze_code(current_code)
                    if estimators:
                        RuntimeOracle.log_execution(
                            estimators[0].name, n_rows, m_cols, 
                            estimators[0].params, exec_result.execution_time_ms / 1000.0
                        )
                finally:
                    poller.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await poller
                        
                    # Final flush
                    if task_id:
                        # Send a final update to ensure UI sees the end state before COMPLETE event
                        pass

                if exec_result.error is None:
                    print(f"    [Executor] ✅ Execution successful.", flush=True)
                    if exec_result.kernel_state:
                        self.authoritative_state.update(exec_result.kernel_state)
                        print(f"    [Executor] Memory updated. Total variables: {len(self.authoritative_state)}", flush=True)
                    # Step succeeded after ≥1 failure → record that its prior errors were
                    # recoverable (distinguishes recurring-but-fixable from hard dead ends).
                    if error_history:
                        record_resolution(recipe_id, recipe_version, task_description,
                                           coder_res.model_used)
                    return exec_result, coder_res.model_used
                else:
                    ename = exec_result.error.get("ename", "Error")
                    evalue = exec_result.error.get("evalue", "Unknown error")
                    print(f"    [Executor] ❌ Failure: {ename} - {evalue}", flush=True)

                    attempt_msg = f"{ename}: {evalue}"

                    # Enrich KeyError feedback with schema hints when columns are missing
                    # from the DataFrame being operated on but exist in a known parquet file.
                    if ename in ("KeyError", "ValueError") and "not in index" in evalue and self.file_schemas:
                        missing_cols = [c.strip().strip("'\"") for c in re.findall(r"['\"]([^'\"]+)['\"]", evalue)]
                        hints = []
                        for fname, schema in self.file_schemas.items():
                            if not isinstance(schema, dict): continue
                            found = [c for c in missing_cols if c in schema]
                            if found:
                                hints.append(f"  - '{fname}' contains columns: {found}")
                        if hints:
                            attempt_msg += (
                                "\n\nHINT: The missing columns exist in separate parquet files. "
                                "You MUST load each parquet file and merge on 'id' before selecting these columns:\n"
                                + "\n".join(hints)
                                + "\nUse: df = df1.merge(df2, on='id').merge(df3, on='id') etc."
                            )
                            print(f"    [Executor] Added schema hints for {len(hints)} parquet file(s)", flush=True)

                    # Record this attempt and rebuild the cumulative feedback: the Coder
                    # sees EVERY prior error (most recent last), so it can avoid all the
                    # dead ends it has already hit, not just the latest one.
                    error_history.append(attempt_msg)
                    error_feedback = "\n".join(
                        f"  Attempt {i + 1} — {_truncate_error_msg(m)}"
                        for i, m in enumerate(error_history)
                    )
                    previous_code = current_code

                    # Persist to the cross-run error ledger (recipe-scoped, structural only)
                    # so this failure informs future first-attempt priors + hardening.
                    record_error(recipe_id, recipe_version, task_description, ename, evalue,
                                 self.coder.model)

                    # Same-reason guard: if this failure reason has now occurred twice, the
                    # model is looping rather than progressing — stop retrying this task.
                    reason = normalize_error_reason(ename, evalue)
                    error_reason_counts[reason] = error_reason_counts.get(reason, 0) + 1
                    if error_reason_counts[reason] >= 2:
                        print(f"    [Executor] 🛑 Same failure reason twice ({ename}) — "
                              f"stopping retries after {retry_count + 1} attempt(s); no progress.", flush=True)
                        break
                    retry_count += 1

            except CodeGenerationError as e:
                # Nothing ran: the model emitted no parseable program (reasoning models
                # leak deliberation prose when they never open a fence). Returning here
                # would abort the task on a generation hiccup, and feeding the raw prose
                # onward produces a SyntaxError the model cannot act on — so treat it as
                # a normal failed attempt whose feedback carries the actual remedy.
                print(f"    [Executor] ❌ No usable code generated: {e}", flush=True)
                # Keep the offending text visible: without it these failures are
                # undebuggable — nothing reaches the sandbox, so no stdout/stderr and
                # no result_json.code is ever written for the attempt.
                _bad = getattr(e, "text", "") or ""
                if _bad:
                    print(f"    [Executor] ┌ generated text ({len(_bad)} chars), first 500:\n"
                          + "\n".join("    │ " + ln for ln in _bad[:500].splitlines())
                          + "\n    └", flush=True)
                    last_bad_text = _bad
                attempt_msg = (
                    f"CodeGenerationError: {e}\n\n"
                    "REMEDY: Output ONLY a single ```python fenced block containing the "
                    "complete program. Do NOT deliberate in prose, do NOT restate the task, "
                    "and do NOT think step-by-step outside the fence — write the code directly."
                )
                if getattr(e, "truncated", False):
                    attempt_msg += (
                        " Your last generation was cut off at the token budget: keep the "
                        "program short and skip commentary."
                    )
                error_history.append(attempt_msg)
                error_feedback = "\n".join(
                    f"  Attempt {i + 1} — {_truncate_error_msg(m)}"
                    for i, m in enumerate(error_history)
                )
                record_error(recipe_id, recipe_version, task_description,
                             "CodeGenerationError", str(e), self.coder.model)
                reason = normalize_error_reason("CodeGenerationError", str(e))
                error_reason_counts[reason] = error_reason_counts.get(reason, 0) + 1
                if error_reason_counts[reason] >= 2:
                    print(f"    [Executor] 🛑 Same failure reason twice (CodeGenerationError) — "
                          f"stopping retries after {retry_count + 1} attempt(s); no progress.", flush=True)
                    exec_result = ExecutionResult(
                        stdout="", stderr=last_bad_text[:8000],
                        error={"ename": "CodeGenerationError", "evalue": str(e)},
                        execution_time_ms=0, kernel_state={}
                    )
                    break
                retry_count += 1
                continue
            except asyncio.TimeoutError:
                print(f"    [Executor] ❌ Timeout (LLM or sandbox execution)", flush=True)
                return ExecutionResult(
                    stdout="", stderr="",
                    error={"ename": "TimeoutError", "evalue": "Operation timed out (LLM generation or sandbox execution exceeded limit)"},
                    execution_time_ms=600000,
                    kernel_state={}
                ), self.coder.model
            except Exception as e:
                print(f"    [Executor] ❌ Unexpected error: {e}", flush=True)
                import traceback
                traceback.print_exc()
                return ExecutionResult(
                    stdout="", stderr="",
                    error={"ename": "RuntimeError", "evalue": str(e)},
                    execution_time_ms=0,
                    kernel_state={}
                ), self.coder.model

        # NATIVE FALLBACK (opt-in, post-exhaustion): the model ran and exhausted its retries.
        # If this recipe node declares a native safety net, invoke it deterministically in the
        # live kernel — one call, no replan. The model's capability was still MEASURED (it
        # tried first and failed); this only recovers the run. See approach_docs/019.
        if fallback_native and fallback_call and fallback_mode in ("native", "native_then_cloud"):
            fb_result = await self._run_native_fallback(
                fallback_native, fallback_call, project_id, session_id)
            if fb_result is not None:
                if error_history:
                    record_resolution(recipe_id, recipe_version, task_description,
                                      f"native_fallback:{fallback_native}")
                return fb_result, f"native_fallback:{fallback_native}"

        return exec_result, self.coder.model

    async def _run_native_fallback(self, fallback_native, fallback_call, project_id, session_id):
        """Inject one native's source into the live kernel and run its canonical call to
        satisfy a node whose model codegen exhausted retries. Returns a successful
        ExecutionResult, or None if the native is unavailable / the fallback itself errored.
        Best-effort: never raises."""
        try:
            from gads.knowledge.native import NATIVE_SOURCE
        except Exception:
            NATIVE_SOURCE = {}
        if fallback_native not in NATIVE_SOURCE:
            print(f"    [Executor] ⚠ No native source for fallback '{fallback_native}'; "
                  "failing normally.", flush=True)
            return None
        print(f"    [Executor] ⛑ Native fallback: model exhausted retries — invoking "
              f"'{fallback_native}' for this node.", flush=True)
        fb_preamble = (
            "import warnings as _wfb\n_wfb.filterwarnings('ignore')\n"
            + NATIVE_SOURCE[fallback_native] + "\n"
            "if '_gads_insights' not in globals(): _gads_insights = []\n"
            "def gads_emit_insight(artifact, insight, evidence=''):\n"
            "    _gads_insights.append({'artifact': artifact, 'insight': insight, 'evidence': evidence})\n"
        )
        fb_postamble = ("\nimport json as _jfb\n"
                        "print('GADS_INSIGHTS_JSON:' + _jfb.dumps(_gads_insights))\n"
                        "_gads_insights = []\n")
        fb_code = fb_preamble + "\n" + fallback_call + "\n" + fb_postamble
        try:
            fb_result = await asyncio.wait_for(
                self.sandbox.execute(fb_code, project_id=project_id, session_id=session_id),
                timeout=360.0)
        except Exception as e:
            print(f"    [Executor] ⚠ Native fallback '{fallback_native}' raised: {e}", flush=True)
            return None
        fb_result.code = fallback_call
        if fb_result.error is not None:
            print(f"    [Executor] ⚠ Native fallback '{fallback_native}' also failed: "
                  f"{fb_result.error.get('evalue', '')[:120]}", flush=True)
            return None
        if "GADS_INSIGHTS_JSON:" in fb_result.stdout:
            try:
                parts = fb_result.stdout.split("GADS_INSIGHTS_JSON:")
                fb_result.semantic_insights = json.loads(parts[1].strip().split("\n")[0])
                fb_result.stdout = parts[0] + "\n".join(parts[1].strip().split("\n")[1:])
            except Exception:
                pass
        if fb_result.kernel_state:
            self.authoritative_state.update(fb_result.kernel_state)
        print(f"    [Executor] ✅ Native fallback '{fallback_native}' succeeded.", flush=True)
        return fb_result
