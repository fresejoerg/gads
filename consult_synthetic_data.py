import json
import urllib.request

def consult_claude():
    prompt = """
    I am refining my Multi-Agent Data Science system (GADS).
    I previously implemented a 'Hallucination Guard' that blocked agents from creating hardcoded DataFrames (using AST validation).
    
    The user wants to restore the ability to generate synthetic data, but ensure it is used CONSISTENTLY downstream.
    
    PROBLEM:
    In a previous run, Task 1 generated a 'flower' dataframe (but actually filled it with AI definitions). 
    Task 2 was supposed to summarize it, but instead of summarizing the actual data in memory (AI definitions), 
    it hallucinated a story about flowers.
    
    PROPOSED SOLUTION:
    1. Relax the AST guard to allow synthetic data.
    2. Implement 'State Inspection': After Task 1, the system inspects the sandbox memory (e.g., `df.head()`).
    3. Inject 'Memory Context': Before Task 2, the system tells the Coder: "Variable 'df' actually contains: [Sample Data]".
    
    QUESTION:
    1. What is the most robust way to 'inspect' a stateful IPython kernel's variables to provide this context to the next agent?
    2. How can I prompt the agent to prioritize 'Variables in Memory' over the 'General Task Description' to prevent it from ignoring the actual data?
    """

    url = "http://localhost:4000/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-1234"
    }
    data = {
        "model": "claude-opus-4.7",
        "messages": [{"role": "user", "content": prompt}]
    }

    req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            print(res_data['choices'][0]['message']['content'])
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    consult_claude()
