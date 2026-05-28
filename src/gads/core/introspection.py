import os
import json
import base64
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def harden_json_artifact(file_path: str):
    """
    Deterministically decodes Plotly 'bdata' (binary encoding) and re-saves
    as standard, browser-compatible JSON. This is the definitive server-side fix.
    """
    if not os.path.exists(file_path): return
    
    def decode_bdata_recursive(obj):
        if isinstance(obj, dict):
            if "bdata" in obj and "dtype" in obj:
                try:
                    raw = base64.b64decode(obj["bdata"])
                    # Map Plotly/Numpy dtypes to standard ones
                    dtype_map = {
                        "i1": "int8", "i2": "int16", "i4": "int32", "i8": "int64",
                        "u1": "uint8", "u2": "uint16", "u4": "uint32", "u8": "uint64",
                        "f4": "float32", "f8": "float64"
                    }
                    dt = dtype_map.get(obj["dtype"], "float64")
                    arr = np.frombuffer(raw, dtype=dt)
                    if "shape" in obj:
                        shape = tuple(int(s.strip()) for s in obj["shape"].split(","))
                        arr = arr.reshape(shape)
                    return arr.tolist()
                except Exception:
                    return obj # Fallback to original if decoding fails
            return {k: decode_bdata_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [decode_bdata_recursive(x) for x in obj]
        elif isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        return obj

    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Check if it looks like Plotly data
        if not (isinstance(data, dict) and "data" in data):
            return # Not a plotly chart, skip hardening

        # Perform deep decode
        clean_data = decode_bdata_recursive(data)
        
        # Save back to disk
        with open(file_path, "w") as f:
            json.dump(clean_data, f)
            
        print(f"  [Introspection] Hardened Plotly JSON (Binary Purged): {os.path.basename(file_path)}")
    except Exception as e:
        print(f"  [Introspection] Warning: Failed to harden {file_path}: {e}")

def summarize_artifact(file_path: str) -> str:
    """
    Computes a deterministic, human-readable (and LLM-friendly) summary of an artifact.
    Does NOT use LLMs. This is ground truth.
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"

    fname = os.path.basename(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == ".json":
            # Check if it's a Plotly JSON or just data
            with open(file_path, "r") as f:
                try:
                    data = json.load(f)
                except Exception:
                    return f"[{fname}] Error: Invalid JSON at {file_path}"
            
            # Plotly signature: has 'data' key as a list
            if isinstance(data, dict) and "data" in data and isinstance(data["data"], list) and len(data["data"]) > 0:
                chart_type = data["data"][0].get("type", "unknown")
                trace_names = [t.get("name", "unnamed") for t in data["data"]]
                title = data.get("layout", {}).get("title", {}).get("text", "Untitled")
                return f"[{fname}] Plotly {chart_type.capitalize()} Chart: '{title}'. Traces: {', '.join(trace_names[:5])}."
            else:
                # Generic JSON data — surface scalar values when present
                size_kb = os.path.getsize(file_path) / 1024
                if isinstance(data, dict) and size_kb < 10:
                    scalar_items = [(k, v) for k, v in data.items() if isinstance(v, (int, float, bool, str))]
                    if scalar_items:
                        parts = [f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in scalar_items[:10]]
                        return f"[{fname}] JSON Data ({size_kb:.1f} KB): {', '.join(parts)}."
                keys = list(data.keys()) if isinstance(data, dict) else []
                return f"[{fname}] JSON Data File: {size_kb:.1f} KB. Keys: {', '.join(keys[:10])}."

        elif ext in (".csv", ".parquet"):
            if ext == ".csv":
                df = pd.read_csv(file_path, nrows=100)
                # Get actual size for CSV
                rows = sum(1 for _ in open(file_path)) - 1
            else:
                df = pd.read_parquet(file_path)
                rows = len(df)
            
            cols = list(df.columns)
            return f"[{fname}] Dataset ({ext[1:].upper()}): {rows} rows, {len(cols)} columns. Schema: {', '.join(cols[:15])}{'...' if len(cols) > 15 else ''}."

        elif ext in (".png", ".jpg", ".jpeg"):
            size_kb = os.path.getsize(file_path) / 1024
            return f"[{fname}] Static Image ({ext[1:].upper()}): {size_kb:.1f} KB."

        else:
            return f"[{fname}] File ({ext[1:].upper()}): {os.path.getsize(file_path)} bytes."

    except Exception as e:
        return f"[{fname}] Error summarizing: {str(e)}"
