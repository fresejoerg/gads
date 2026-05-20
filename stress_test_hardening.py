import asyncio
import httpx
import json
import os

async def main():
    plan_path = "/home/joergf/.gemini/tmp/gads/306a369b-734f-4e91-9a39-cc777767baea/plans/context-hardening-approach.md"
    with open(plan_path, "r") as f:
        plan_content = f.read()

    system_prompt = """
You are Claude 4.7 Opus, a world-class AI Architect and Data Science Strategist.
Your goal is to stress-test and brutally critique a proposed plan for "Context Hardening" in an automated multi-agent data science system.
The system is suffering from context overflow (56k tokens) and over-strict agent audits.
Focus on:
1. The feasibility of "Semantic Summaries" vs full artifact data. Can an LLM actually synthesize well from just metadata?
2. The risks of "Critique Softening." Does this lead to silent quality decay?
3. The technical complexity of the "Sliding Window" in a stateful kernel environment.
4. XSS and security when injecting distilled Markdown into prompts.
"""
    user_content = f"### THE PLAN TO CRITIQUE:\n\n{plan_content}\n\nPlease provide a structured critique highlighting any technical debt, edge cases, or logic flaws in this architecture."

    print("--- Sending plan to Claude Opus 4.7 for stress-test ---")
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
