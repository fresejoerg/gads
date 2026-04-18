import asyncio
import os
import sys

# Ensure src directory is in path
sys.path.append(os.path.join(os.getcwd(), "src"))

from gads.agents.planner import DataSciencePlanner, PlannerInput
from gads.agents.workers.nlp import NLPExtractorAgent, NLPExtractorInput

async def main():
    print("--- 🚀 GADS: Data Science Rockstar Initializing ---")
    
    # 1. Initialize Agents
    planner = DataSciencePlanner()
    nlp_worker = NLPExtractorAgent()
    
    # 2. Project Input
    objective = "Extract all organizations and dates from the following text: 'Anthropic released Claude 4.7 on April 16, 2026. Google followed with Gemini 3.1 shortly after.'"
    
    print(f"\n[Project Manager] Objective: {objective}")
    
    # 3. Planning Phase (Uses strong model)
    print("\n[Planning] Generating execution DAG...")
    plan_output = await planner.run(PlannerInput(objective=objective))
    
    for i, task in enumerate(plan_output.steps):
        print(f"  Step {i+1}: {task.description} (Assignee: {task.assigned_to})")
    
    # 4. Execution Phase (Uses local model for worker)
    print("\n[Execution] Starting Worker tasks...")
    # For this demo, we just call the worker directly based on the objective
    worker_result = await nlp_worker.run(NLPExtractorInput(text=objective))
    
    print("\n[Result] Extracted Entities:")
    for entity in worker_result.entities:
        print(f"  - {entity.name} ({entity.category}): {entity.context}")

if __name__ == "__main__":
    asyncio.run(main())
