import ast
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel

class EstimatorInfo(BaseModel):
    name: str
    params: Dict[str, Any]
    complexity: str # 'linear', 'nlogn', 'quadratic'

class RuntimeOracle:
    """Predicts execution time based on AST analysis and historical data."""
    
    COMPLEXITY_FACTORS = {
        'linear': 1.0,
        'nlogn': 2.5,
        'quadratic': 10.0,
        'unknown': 5.0
    }
    
    # Heuristic for hardware (seconds per N*M units)
    # Calibrated for a basic CPU. Will be refined by ExecutionLogs.
    BASE_HARDWARE_CONSTANT = 1_000_000 # Higher = faster

    @staticmethod
    def analyze_code(code: str) -> List[EstimatorInfo]:
        """Scans code for known estimators and their hyperparams."""
        estimators = []
        try:
            tree = ast.parse(code)
        except Exception:
            return []

        # Map to resolve imports
        imports = {} # alias -> full_name

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    full_name = f"{node.module}.{alias.name}"
                    imports[alias.asname or alias.name] = full_name

            # Look for class instantiations
            if isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    # Handle cases like 'sklearn.ensemble.RandomForestClassifier'
                    if isinstance(node.func.value, ast.Name):
                        func_name = f"{node.func.value.id}.{node.func.attr}"
                
                if func_name and (func_name in imports or any(k in func_name for k in ['Classifier', 'Regressor', 'SVC', 'SVR', 'XGB', 'LGBM'])):
                    full_class_name = imports.get(func_name, func_name)
                    
                    # Extract keyword arguments
                    params = {}
                    for kw in node.keywords:
                        if isinstance(kw.value, (ast.Constant, ast.Num, ast.Str)):
                            params[kw.arg] = getattr(kw.value, 'value', getattr(kw.value, 'n', getattr(kw.value, 's', None)))
                    
                    complexity = 'linear'
                    if any(k in full_class_name for k in ['RandomForest', 'XGB', 'LGBM', 'GradientBoosting', 'Tree']):
                        complexity = 'nlogn'
                    elif any(k in full_class_name for k in ['SVC', 'SVR', 'KNeighbors']):
                        complexity = 'quadratic'
                    
                    estimators.append(EstimatorInfo(
                        name=full_class_name,
                        params=params,
                        complexity=complexity
                    ))
        
        return estimators

    # Sentence-transformer: ~10s per 1000 rows on CPU (conservative estimate)
    EMBEDDING_SECONDS_PER_KROW = 15.0

    @classmethod
    def _detect_embedding(cls, code: str) -> bool:
        """Returns True if code uses sentence-transformers encode() on a large corpus."""
        embedding_markers = ["SentenceTransformer", "sentence_transformers", ".encode("]
        return any(m in code for m in embedding_markers)

    @classmethod
    def estimate_runtime(cls, code: str, n_rows: int, m_cols: int) -> float:
        """
        Returns estimated runtime in seconds.
        T = (N * log(N) * M * complexity_factor * multipliers) / HardwareConstant
        Sentence-transformer encoding is estimated separately: ~15s / 1000 rows.
        """
        import math

        # Sentence-transformer short-circuit: bypass before OOM/timeout
        if cls._detect_embedding(code) and n_rows > 5000:
            est = (n_rows / 1000.0) * cls.EMBEDDING_SECONDS_PER_KROW
            print(f"    [Oracle] Sentence-transformer detected on {n_rows} rows → {est:.0f}s estimate")
            return est

        estimators = cls.analyze_code(code)
        if not estimators:
            return 1.0 # Default low estimate for simple code

        max_est = 0.0
        for est in estimators:
            factor = cls.COMPLEXITY_FACTORS.get(est.complexity, 5.0)
            
            # Dimensional multipliers
            n = n_rows or 1
            m = m_cols or 1
            
            if est.complexity == 'linear':
                units = n * m
            elif est.complexity == 'nlogn':
                units = n * math.log2(max(n, 2)) * m
            else: # quadratic
                units = (n ** 2) * m
            
            # Hyperparameter multipliers
            k = 1.0
            if 'n_estimators' in est.params:
                k *= (int(est.params['n_estimators']) / 100.0) # Default RF is 100
            if 'cv' in est.params:
                k *= int(est.params['cv'])
                
            est_time = (units * factor * k) / cls.BASE_HARDWARE_CONSTANT
            max_est = max(max_est, est_time)
            
        return max_est

    @staticmethod
    def log_execution(estimator: str, n: int, m: int, params: Dict, runtime: float):
        """Records actual performance to the DB for future learning."""
        from gads.core.database import engine
        from gads.core.models import ExecutionLog
        from sqlmodel import Session
        
        with Session(engine) as session:
            log = ExecutionLog(
                estimator_class=estimator,
                n_rows=n,
                m_cols=m,
                params_json=params,
                hardware_id="local_sandbox",
                actual_runtime_s=runtime
            )
            session.add(log)
            session.commit()
