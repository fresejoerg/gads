import asyncio
import os
import sys

# Ensure src directory is in path
sys.path.append(os.path.join(os.getcwd(), "src"))

from gads.core.state import Blackboard
from gads.core.executor import ExecutionManager

async def main():
    print("--- 🚀 GADS: Data Science Rockstar - Stateful Sandbox Demo ---")
    
    # 1. Initialize Blackboard and Execution Manager
    blackboard = Blackboard(
        project_name="Sandbox Verification",
        objective="Create data, analyze it, and plot it statefully."
    )
    executor = ExecutionManager()
    session_id = "verification-session-001"

    # 2. Task 1: Data Creation
    print("\n[Task 1] Creating a synthetic dataset of 100 rows...")
    res1 = await executor.run_task(
        "Create a pandas DataFrame 'df' with 100 rows of random data. Columns: 'age' (int 20-60), 'salary' (float).",
        blackboard,
        session_id=session_id
    )
    print(f"  Result: {res1.stdout}")

    # 3. Task 2: Data Analysis (Stateful)
    print("\n[Task 2] Calculating mean salary (reusing 'df')...")
    res2 = await executor.run_task(
        "Calculate the mean of the 'salary' column from 'df' and print it.",
        blackboard,
        session_id=session_id
    )
    print(f"  Result: {res2.stdout}")

    # 4. Task 3: Data Visualization (Stateful)
    print("\n[Task 3] Generating a plot...")
    res3 = await executor.run_task(
        "Create a histogram of the 'age' column from 'df'. Use seaborn.",
        blackboard,
        session_id=session_id
    )
    print(f"  Number of plots captured: {len(res3.plots)}")
    if res3.plots:
        print(f"  Success: Plot captured (Base64 length: {len(res3.plots[0])})")

    await executor.sandbox.close()

if __name__ == "__main__":
    asyncio.run(main())
