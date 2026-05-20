import asyncio
import httpx
import json
import os

async def main():
    plan_path = "approach_docs/temporal_integration.md"
    with open(plan_path, "r") as f:
        plan_content = f.read()

    system_prompt = """
You are Claude 4.7 Opus, a world-class Distributed Systems Architect and AI Orchestration Expert.
Your goal is to stress-test and brutally critique a proposed plan for integrating **Temporal** and **Pydantic AI** into an existing multi-agent Data Science system (GADS).
The system manages sandboxed Python environments, expensive LLM calls, and complex multi-step plans.
Focus on:
1. **The "Replay" Trap**: Temporal workflows are deterministic and work via replay. How does this architecture handle the *side effects* in the remote sandbox (e.g., creating files, model training)? If a workflow crashes and replays, will it try to re-run the same model training task in a sandbox that already has the model file?
2. **State Synchronization**: The plan keeps SQLModel for the UI and Temporal for the workflow logic. How do we ensure the UI doesn't show stale state if a Temporal Activity fails but hasn't yet updated the DB?
3. **Model Tiering & Latency**: Does moving every LLM call into a Temporal Activity (via TemporalAgent) add significant overhead that could frustrate an interactive user?
4. **Complexity vs. Value**: Is Temporal overkill for a system that mostly runs sequential worker tasks? What is the *actual* technical debt being bought here?
"""
    user_content = f"### THE TEMPORAL REFACTORING PLAN:\n\n{plan_content}\n\nPlease provide a structured critique focusing on state safety, side-effect management, and the feasibility of using TemporalAgent for long-running sandboxed tasks."

    print("--- Sending Temporal Plan to Claude Opus 4.7 for stress-test ---")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://localhost:4000/v1/chat/completions",
                headers={"Authorization": "Bearer sk-1234", "Content-Type": "application/json"},
                json={
                    "model": "claude-opus-4.7",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ]
                },
                timeout=120.0
            )
            response.raise_for_status()
            result = response.json()
            print("\n### CRITIQUE RECEIVED:")
            print(result['choices'][0]['message']['content'])
        except Exception as e:
            print(f"Stress-test failed: {e}")
            if isinstance(e, httpx.HTTPStatusError):
                print(e.response.text)

if __name__ == "__main__":
    asyncio.run(main())
