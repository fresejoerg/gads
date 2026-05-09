import os
import uuid
import zipfile
import json
from typing import List, Dict, Any, Optional
from gads.tools.sandbox import SandboxClient

class HandoverManager:
    """Handles the creation of reproducible project bundles for bypassed tasks."""

    def __init__(self, sandbox_url: str = "http://localhost:8000"):
        self.sandbox = SandboxClient(base_url=sandbox_url)

    async def create_bundle(
        self, 
        project_id: uuid.UUID, 
        heavy_code: str, 
        estimated_runtime: float,
        workspace_root: str = "/home/joergf/projects/MyLocalStack/data/workspaces"
    ) -> str:
        """
        1. Exports in-memory state to files.
        2. Generates a standalone script.
        3. Creates a ZIP archive.
        Returns the filename of the generated ZIP.
        """
        project_dir = os.path.join(workspace_root, str(project_id))
        bundle_name = f"handover_{project_id.hex[:8]}.zip"
        bundle_path = os.path.join(project_dir, bundle_name)
        
        # 1. Export State from Sandbox
        # We ask the sandbox to save all 'DataFrame' and 'list/dict' variables to disk.
        export_code = """
import pandas as pd
import pickle
import os
import json

exported_files = []
for name, val in list(globals().items()):
    if name.startswith('_') or callable(val):
        continue
    try:
        if isinstance(val, pd.DataFrame):
            fname = f"state_{name}.parquet"
            val.to_parquet(fname)
            exported_files.append({"name": name, "file": fname, "type": "dataframe"})
        elif isinstance(val, (list, dict)):
            fname = f"state_{name}.pkl"
            with open(fname, 'wb') as f:
                pickle.dump(val, f)
            exported_files.append({"name": name, "file": fname, "type": "pickle"})
    except: pass

print(json.dumps({"exported": exported_files}))
"""
        res = await self.sandbox.execute(export_code, project_id=project_id, session_id=str(project_id))
        
        exported_metadata = []
        if res.stdout:
            try:
                data = json.loads(res.stdout.strip().split('\n')[-1])
                exported_metadata = data.get("exported", [])
            except: pass

        # 2. Generate Standalone Script
        script_content = self._generate_script(heavy_code, exported_metadata, estimated_runtime)
        script_name = "run_model_training.py"
        script_path = os.path.join(project_dir, script_name)
        with open(script_path, "w") as f:
            f.write(script_content)

        # 3. Create ZIP
        with zipfile.ZipFile(bundle_path, 'w') as zipf:
            # Add the generated script
            zipf.write(script_path, arcname=script_name)
            
            # Add exported state files
            for item in exported_metadata:
                fpath = os.path.join(project_dir, item["file"])
                if os.path.exists(fpath):
                    zipf.write(fpath, arcname=item["file"])
            
            # Add existing workspace files (CSVs etc)
            for f in os.listdir(project_dir):
                if f.endswith(('.csv', '.parquet')) and not f.startswith('state_'):
                    zipf.write(os.path.join(project_dir, f), arcname=f)
            
            # Add a basic README
            readme = f"""# GADS Handover Bundle
Project: {project_id}
Estimated Runtime: {estimated_runtime/60:.1f} minutes

## Instructions
1. Ensure you have the required dependencies installed:
   `pip install pandas scikit-learn pyarrow fastparquet`
2. Run the training script:
   `python {script_name}`

This bundle was generated because the task exceeded the 5-minute online safety limit.
"""
            zipf.writestr("README.md", readme)

        return bundle_name

    def _generate_script(self, heavy_code: str, metadata: List[Dict], runtime: float) -> str:
        """Templates a standalone script that loads state first."""
        loader_code = "import pandas as pd\nimport pickle\nimport os\n\nprint('--- Loading GADS State Artifacts ---')\n"
        for item in metadata:
            if item["type"] == "dataframe":
                loader_code += f"{item['name']} = pd.read_parquet('{item['file']}')\n"
            else:
                loader_code += f"with open('{item['file']}', 'rb') as f: {item['name']} = pickle.load(f)\n"
        
        full_script = f"""# GADS Standalone Execution Script
# Estimated completion: {runtime/60:.1f} minutes

{loader_code}

print('\\n--- Starting Heavy Computation ---')
import time
start = time.time()

{heavy_code}

print(f'\\n✅ Task Complete in {{time.time() - start:.2f}}s')
"""
        return full_script
