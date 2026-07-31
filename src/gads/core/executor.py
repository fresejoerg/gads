import asyncio
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
    if _HAS_FILE_LOAD and _HAS_TRAINING and not _HAS_SAMPLE:
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

    return code


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
        recipe_version: Optional[str] = None
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
        max_attempts = 10
        retry_count = 0
        error_feedback = None
        previous_code = ""
        error_history = []           # one human-readable error string per failed attempt
        error_reason_counts = {}     # normalized reason -> count (a reason seen twice => stop)

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

                # Inject AutoGluon native node preamble when the code references AutoGluon APIs
                _autogluon_preamble = ""
                if any(kw in current_code for kw in ["autogluon", "TabularPredictor", "TimeSeriesPredictor",
                                                       "gads_automl_fit", "gads_timeseries_fit", "gads_calibrate_threshold"]):
                    try:
                        from gads.knowledge.native import AUTOGLUON_PREAMBLE
                        _autogluon_preamble = AUTOGLUON_PREAMBLE + "\n"
                        print(f"    [Executor] Injecting AutoGluon native node preamble", flush=True)
                    except Exception as _e:
                        print(f"    [Executor] Warning: Could not load AutoGluon preamble: {_e}", flush=True)

                # Inject causal native node preamble when the code references causal APIs
                _causal_preamble = ""
                if any(kw in current_code for kw in ["CausalModel", "dowhy", "causal_estimate",
                                                       "gads_causal_estimate_ate", "gads_causal_bayesian_ate",
                                                       "bambi", "bmb.Model"]):
                    try:
                        from gads.knowledge.native import CAUSAL_PREAMBLE
                        _causal_preamble = CAUSAL_PREAMBLE + "\n"
                        print(f"    [Executor] Injecting causal native node preamble", flush=True)
                    except Exception as _e:
                        print(f"    [Executor] Warning: Could not load causal preamble: {_e}", flush=True)

                # Inject recommendation native node preamble when CF/recommender APIs are referenced
                _rec_preamble = ""
                if any(kw in current_code for kw in ["gads_recommend_and_evaluate", "gads_build_interaction_matrix",
                                                       "gads_fit_and_recommend", "gads_evaluate_topn",
                                                       "gads_temporal_loo_split", "AlternatingLeastSquares",
                                                       "implicit.als"]):
                    try:
                        from gads.knowledge.native import RECOMMENDATION_PREAMBLE
                        _rec_preamble = RECOMMENDATION_PREAMBLE + "\n"
                        print(f"    [Executor] Injecting recommendation native node preamble", flush=True)
                    except Exception as _e:
                        print(f"    [Executor] Warning: Could not load recommendation preamble: {_e}", flush=True)

                # Inject model-audit native node preamble when the skore audit is referenced
                _audit_preamble = ""
                if any(kw in current_code for kw in ["gads_audit_model", "EstimatorReport", "skore"]):
                    try:
                        from gads.knowledge.native import MODEL_AUDIT_PREAMBLE
                        _audit_preamble = MODEL_AUDIT_PREAMBLE + "\n"
                        print(f"    [Executor] Injecting model-audit native node preamble", flush=True)
                    except Exception as _e:
                        print(f"    [Executor] Warning: Could not load model-audit preamble: {_e}", flush=True)

                # Inject survival-analysis native node preamble when survival APIs are referenced
                _survival_preamble = ""
                if any(kw in current_code for kw in ["gads_make_surv_target", "gads_evaluate_survival",
                                                       "gads_cox_ph_report", "gads_kaplan_meier",
                                                       "CoxPHFitter", "CoxPHSurvivalAnalysis",
                                                       "RandomSurvivalForest", "lifelines", "sksurv",
                                                       "Surv.from_", "KaplanMeierFitter"]):
                    try:
                        from gads.knowledge.native import SURVIVAL_PREAMBLE
                        _survival_preamble = SURVIVAL_PREAMBLE + "\n"
                        print(f"    [Executor] Injecting survival-analysis native node preamble", flush=True)
                    except Exception as _e:
                        print(f"    [Executor] Warning: Could not load survival preamble: {_e}", flush=True)

                # Wrap code with telemetry hooks
                telemetry_preamble = _autogluon_preamble + _causal_preamble + _rec_preamble + _audit_preamble + _survival_preamble + """
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
                        f"  Attempt {i + 1} — {m}" for i, m in enumerate(error_history)
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

        return exec_result, self.coder.model
