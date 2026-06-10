import asyncio
import uuid
from gads.tools.sandbox import SandboxClient

async def main():
    client = SandboxClient()
    try:
        packages = [
            "dowhy",
            "econml",
            "causalml",
            "causal_learn", # note: import package name is 'causallearn' or 'causal-learn' (let's check both)
            "pymc",
            "arviz",
            "bambi",
            "causalimpact", # pycausalimpact is imported as 'causalimpact'
            "pgmpy",
            "causalinference", # CausalInference is imported as 'causalinference'
            "autogluon.tabular",
            "autogluon.timeseries"
        ]
        
        # Test each package import individually
        for pkg in packages:
            code = f"import {pkg}\nprint('{pkg} imported successfully')"
            if pkg == "causal_learn":
                code = "import causallearn\nprint('causallearn imported successfully')"
            
            res = await client.execute(code, project_id=uuid.uuid4(), session_id="verify_test")
            if res.error:
                print(f"❌ Failed to import {pkg}: {res.error['evalue']}")
                if res.stderr:
                    print(f"   Stderr: {res.stderr}")
            else:
                print(f"✅ {pkg}: {res.stdout.strip()}")
                
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())
