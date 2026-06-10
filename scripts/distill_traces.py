#!/usr/bin/env python3
import os
import re
import yaml
import json
import argparse
import dotenv
from typing import List, Dict, Any, Optional
from langfuse import Langfuse
from gads.core.knowledge import KnowledgeRegistry, Recipe

# Load environment variables
dotenv.load_dotenv()

# Explicitly map legacy/external router IDs to current recipe files
RECIPE_MAPPING = {
    "causal_inference_ate_refutation": "causal_effect.observational.dowhy",
    "causal_effect.observational.dowhy": "causal_effect.observational.dowhy",
    "binary_classification.tabular.standard": "binary_classification.tabular.standard",
    "tabular_automl.autogluon.standard": "tabular_automl.autogluon.standard",
}

def clean_words(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words of length >= 3."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    stopwords = {"the", "and", "for", "with", "that", "this", "from", "are", "our", "use", "using"}
    return [w for w in words if w not in stopwords]

def get_word_overlap(text1: str, text2: str) -> int:
    """Calculate overlap score (number of unique common words)."""
    words1 = set(clean_words(text1))
    words2 = set(clean_words(text2))
    return len(words1.intersection(words2))

def extract_task_description(inp: Any) -> Optional[str]:
    """Extract TASK description from CodeGenerator input messages."""
    if not isinstance(inp, dict):
        return None
    messages = inp.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            match = re.search(r"TASK:\s*(.*?)(?:\n\n|$)", content, re.DOTALL)
            if match:
                return match.group(1).strip()
    return None

def extract_code(out: Any) -> Optional[str]:
    """Extract python code from CodeGenerator or Task Execution outputs."""
    if not out:
        return None
    if isinstance(out, dict):
        # CodeGenerator output has content containing JSON
        content = out.get("content", "")
        if content:
            # Try parsing content as JSON
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "code" in data:
                    return data["code"]
            except:
                pass
            # Fallback to looking for code block
            match = re.search(r"```python\s*(.*?)\s*```", content, re.DOTALL)
            if match:
                return match.group(1)
            # Maybe the content is raw python code
            if "import " in content:
                return content
        
        # Task Execution span output has 'code' directly
        if "code" in out:
            return out["code"]
    return None

def get_matched_recipe_id(obs_list: List[Any]) -> Optional[str]:
    """Find matched_recipe_id from Router generation output."""
    for obs in obs_list:
        if obs.name == "Router" and obs.type == "GENERATION" and obs.output:
            out = obs.output
            if isinstance(out, dict):
                content = out.get("content", "")
                try:
                    data = json.loads(content)
                    if isinstance(data, dict) and "matched_recipe_id" in data:
                        return data["matched_recipe_id"]
                except:
                    pass
    return None

def main():
    parser = argparse.ArgumentParser(description="Distill traces and propose recipe invariants.")
    parser.add_argument("--trace-id", type=str, help="Specific Langfuse trace ID to distill.")
    parser.add_argument("--limit", type=int, default=20, help="Number of traces to fetch when trace-id is not specified.")
    args = parser.parse_args()

    # Initialize Langfuse
    pub_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    sec_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST")
    if not pub_key or not sec_key:
        print("Error: LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set in environment.")
        return

    langfuse_client = Langfuse(public_key=pub_key, secret_key=sec_key, host=host)

    # Load Recipe Registry
    registry = KnowledgeRegistry("src/gads/knowledge/recipes")
    print(f"Loaded {len(registry.recipes)} recipes from registry.")

    traces_to_process = []
    if args.trace_id:
        try:
            trace = langfuse_client.get_trace(args.trace_id)
            traces_to_process.append(trace)
        except Exception as e:
            print(f"Error retrieving trace {args.trace_id}: {e}")
            return
    else:
        print(f"Fetching recent {args.limit} traces...")
        try:
            traces_res = langfuse_client.get_traces(limit=args.limit)
            # Filter for completed traces containing Synthesizer stage
            for t in traces_res.data:
                obs = langfuse_client.get_observations(trace_id=t.id)
                # Check if Synthesizer stage ran successfully
                synth_obs = [o for o in obs.data if o.name == "Synthesizer" or o.name.startswith("Synthesis Attempt")]
                if synth_obs:
                    traces_to_process.append(t)
            print(f"Found {len(traces_to_process)} successful/completed traces out of recent {args.limit}.")
        except Exception as e:
            print(f"Error fetching traces: {e}")
            return

    proposals = {}

    for trace in traces_to_process:
        print(f"\nProcessing Trace ID: {trace.id} | Name: {trace.name}")
        obs = langfuse_client.get_observations(trace_id=trace.id)
        
        # Determine matched recipe
        router_recipe_id = get_matched_recipe_id(obs.data)
        mapped_id = RECIPE_MAPPING.get(router_recipe_id) if router_recipe_id else None
        
        recipe: Optional[Recipe] = None
        if mapped_id and mapped_id in registry.recipes:
            recipe = registry.recipes[mapped_id]
        elif router_recipe_id and router_recipe_id in registry.recipes:
            recipe = registry.recipes[router_recipe_id]
        else:
            # Fallback: find recipe by keyword matching against objective
            obj_text = trace.metadata.get("objective", "") if trace.metadata else ""
            best_score = 0
            for r_id, r in registry.recipes.items():
                score = get_word_overlap(obj_text, r.rationale)
                if score > best_score:
                    best_score = score
                    recipe = r
            if recipe:
                print(f"  Fallback routed to recipe: {recipe.id} (score: {best_score})")

        if not recipe:
            print("  Could not resolve a matching recipe for this trace.")
            continue

        print(f"  Matched Recipe: {recipe.id}")

        # Gather successful tasks and extract code
        successful_tasks = []
        code_gens = [o for o in obs.data if o.name == "CodeGenerator" and o.type == "GENERATION"]
        
        for cg in code_gens:
            code = extract_code(cg.output)
            desc = extract_task_description(cg.input)
            if code and desc:
                # Match task description to recipe node intent
                best_node = None
                best_overlap = -1
                for node in recipe.dag:
                    overlap = get_word_overlap(desc, node.intent)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_node = node
                
                successful_tasks.append({
                    "description": desc,
                    "code": code,
                    "matched_node": best_node
                })

        print(f"  Extracted {len(successful_tasks)} successful tasks with code.")

        # Detect and propose invariants
        trace_proposals = []
        obj_text = trace.metadata.get("objective", "") if trace.metadata else ""
        
        # Heuristic 1: Subsampling for CausalModel
        is_causal = "causal" in recipe.id.lower() or "causal" in obj_text.lower()
        has_large_data = any(x in obj_text.lower() for x in ["284,807", "284807", "50000", "50,000", "large", "imbalanced"])
        
        if is_causal:
            # Check if any successful task has subsampled df to 20K
            has_subsample = False
            for task in successful_tasks:
                code_lower = task["code"].lower()
                if ".sample(" in code_lower and ("20000" in code_lower or "20_000" in code_lower):
                    has_subsample = True
                    break
            
            # If large dataset was used but no subsampling or we want to safeguard it
            if has_large_data and not has_subsample:
                inv = "LARGE DATASET PERFORMANCE: for datasets >20K rows, ALWAYS subsample df to 20K before CausalModel construction to prevent timeouts. Use df_sample = df.sample(20000, random_state=42)."
                if inv not in recipe.invariants:
                    trace_proposals.append(inv)

        # Heuristic 2: class_weight='balanced'
        is_classification = "class" in recipe.id.lower() or "classification" in recipe.id.lower() or "classifier" in obj_text.lower()
        if is_classification:
            has_balanced = False
            has_sklearn_clf = False
            for task in successful_tasks:
                code_lower = task["code"].lower()
                if any(clf in code_lower for clf in ["logisticregression", "randomforestclassifier", "histgradientboostingclassifier"]):
                    has_sklearn_clf = True
                    if "class_weight='balanced'" in code_lower or 'class_weight="balanced"' in code_lower:
                        has_balanced = True
                        break
            
            if has_sklearn_clf and not has_balanced:
                inv = "CLASS WEIGHTS: always set class_weight='balanced' in standard scikit-learn classification code patterns (e.g. LogisticRegression, RandomForestClassifier, HistGradientBoostingClassifier) to safeguard against class imbalance."
                if inv not in recipe.invariants:
                    trace_proposals.append(inv)

        # Heuristic 3: random_state=42
        needs_random_state = False
        has_random_state_42 = False
        for task in successful_tasks:
            code_lower = task["code"].lower()
            if any(kw in code_lower for kw in [".sample(", "train_test_split(", "logisticregression(", "randomforestclassifier("]):
                needs_random_state = True
                if "random_state=42" in code_lower or "random_seed=42" in code_lower or "random_state = 42" in code_lower:
                    has_random_state_42 = True
                    break
        
        if needs_random_state and not has_random_state_42:
            inv = "Random seeds must be fixed for reproducibility (random_state=42)."
            if inv not in recipe.invariants:
                trace_proposals.append(inv)

        if trace_proposals:
            proposals[recipe.id] = {
                "filepath": registry._recipe_filepaths[recipe.id],
                "invariants": trace_proposals
            }

    # Write proposals to proposed_invariants.md
    output_path = "proposed_invariants.md"
    
    if proposals:
        with open(output_path, "w") as f:
            f.write("# Proposed Recipe Invariants\n\n")
            f.write("This file contains proposed additions to recipe invariants offline based on trace distillation.\n\n")
            for r_id, prop in proposals.items():
                fn = os.path.basename(prop["filepath"])
                f.write(f"## Recipe: [{fn}](file://{prop['filepath']}) (ID: `{r_id}`)\n\n")
                f.write("```yaml\ninvariants:\n")
                for inv in prop["invariants"]:
                    f.write(f"  - {json.dumps(inv)}\n")
                f.write("```\n\n")
        
        # Print for Human Review Gate (stdout)
        print("\n" + "="*50)
        print("PROPOSED INVARIANTS FOR HUMAN REVIEW")
        print("="*50)
        for r_id, prop in proposals.items():
            print(f"\nRecipe: {r_id} ({os.path.basename(prop['filepath'])})")
            for inv in prop["invariants"]:
                print(f"  - {inv}")
        print("\n" + "="*50)
        print(f"Proposals written successfully to {output_path}")
        print("="*50)
    else:
        print("\nNo new invariants proposed for the processed traces.")
        if os.path.exists(output_path):
            os.remove(output_path)

if __name__ == "__main__":
    main()
