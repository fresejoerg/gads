import os
import dotenv
from langfuse import Langfuse
import json

dotenv.load_dotenv()

def main():
    langfuse_client = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST")
    )
    
    trace_id = "d06f005b-b8cf-49fd-8265-b79695441be2"
    obs = langfuse_client.get_observations(trace_id=trace_id)
    
    code_gens = [o for o in obs.data if o.name == "CodeGenerator" and o.type == "GENERATION"]
    for i, o in enumerate(code_gens):
        out = o.output
        if isinstance(out, dict):
            content = out.get("content", "")
            try:
                val = json.loads(content)
                code = val.get("code", "")
                if ".sample" in code:
                    print(f"\n=== CodeGenerator {i} ID: {o.id} ===")
                    print(f"Code contains .sample:")
                    print(code)
            except:
                if ".sample" in content:
                    print(f"\n=== CodeGenerator {i} ID: {o.id} ===")
                    print(f"Content contains .sample:")
                    print(content)

if __name__ == "__main__":
    main()








