import json
import urllib.request
import os

def consult_claude():
    prompt = """
    I am building a multi-agent Data Science Control Center using Chainlit.
    When I run the app, I get a blank screen (500 Internal Server Error).
    
    ENVIRONMENT:
    - Port 8000: Python Sandbox (FastAPI)
    - Port 8001: GADS Backend (FastAPI + Postgres)
    - Port 8002: Chainlit UI
    - .env contains:
      DATABASE_URL=postgresql://llmuser:llmpassword@localhost:5432/litellm
      GADS_BACKEND_URL=http://localhost:8001
      GADS_WS_URL=ws://localhost:8001/ws
    
    ERROR LOG:
    ```
    ERROR:    Exception in ASGI application
    Traceback (most recent call last):
      ...
      File "/.../chainlit/server.py", line 186, in lifespan
        if data_layer := get_data_layer():
      File "/.../chainlit/data/__init__.py", line 33, in get_data_layer
        from .chainlit_data_layer import ChainlitDataLayer
      File "/.../chainlit/data/chainlit_data_layer.py", line 7, in <module>
        import asyncpg  # type: ignore
    ModuleNotFoundError: No module named 'asyncpg'
    ```
    
    CURRENT UI CODE (src/gads/ui/app.py):
    ```python
    import chainlit as cl
    import httpx
    import json
    import asyncio
    import os
    import websockets
    # ... (standard chainlit handlers)
    ```
    
    OBSERVATION:
    Even after installing `asyncpg`, the 500 error persists.
    
    HYPOTHESIS:
    Chainlit is automatically detecting `DATABASE_URL` and trying to use its own SQL data layer, which might be conflicting with my GADS backend or failing to initialize.
    
    QUESTION:
    1. How do I force Chainlit to IGNORE the DATABASE_URL environment variable so it doesn't try to use its own data layer?
    2. Is there anything else in my ws_listener or handles that looks like a "blind spot" for a 500 error?
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
