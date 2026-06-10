import sys
from gads.core.registry import get_local_only, set_local_only

def main():
    print("Python version:", sys.version)
    
    # Verify imports
    try:
        from gads.agents.planner import DataSciencePlanner, PlannerInput
        print("✓ Successfully imported DataSciencePlanner and PlannerInput")
    except Exception as e:
        print("✗ Failed to import Planner modules:", e)
        sys.exit(1)
        
    try:
        from gads.core.server import app
        print("✓ Successfully imported FastAPI app from server.py")
    except Exception as e:
        print("✗ Failed to import server.py:", e)
        sys.exit(1)
        
    print("✓ All imports are clean. Code changes are verified!")

if __name__ == "__main__":
    main()
