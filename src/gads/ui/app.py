import os
import base64

# --- PRE-IMPORT FIX ---
os.environ.pop("DATABASE_URL", None)

import chainlit as cl
import httpx
import json
import asyncio
import websockets
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("GADS_BACKEND_URL", "http://localhost:8001")
WS_URL = os.getenv("GADS_WS_URL", "ws://localhost:8001/ws")

@cl.on_chat_start
async def start():
    cl.user_session.set("last_seq", 0)
    
    # Initialize the Unified Project Explorer (Files + Memory)
    explorer_el = cl.Text(
        name="📁 Project Explorer", 
        content="### 📁 Workspace Files\n*(No files yet)*\n\n---\n\n### 🧠 Sandbox Memory\n*(Empty)*", 
        display="side"
    )
    
    # Store in session
    cl.user_session.set("explorer_el", explorer_el)
    
    # Send initial message to make the side panel available
    await cl.Message(
        content="--- 🚀 GADS: Data Science Rockstar Control Center ---\n\n"
                "I've initialized your **[📁 Project Explorer]**. "
                "Open the side panel to see your files and memory state update in real-time.",
        elements=[explorer_el]
    ).send()
    
    asyncio.create_task(ws_listener())

async def ws_listener():
    while True:
        try:
            last_seq = cl.user_session.get("last_seq", 0)
            async with websockets.connect(f"{WS_URL}?last_seq={last_seq}") as ws:
                while True:
                    msg = await ws.recv()
                    event = json.loads(msg)
                    await handle_event(event)
                    cl.user_session.set("last_seq", event["seq"])
        except (websockets.ConnectionClosed, ConnectionRefusedError, OSError):
            await asyncio.sleep(2)
        except Exception as e:
            print(f"WS Listener Crash: {e}")
            await asyncio.sleep(5)

async def handle_event(event: dict):
    etype = event["type"]
    payload = event["payload"]
    
    if etype == "STEP_STARTED":
        await cl.Message(content=f"🧠 {payload['message']}").send()

    elif etype == "TASK_CREATED":
        step = cl.Step(name="Task Planned", type="tool")
        step.input = payload['description']
        await step.send()
        cl.user_session.set(f"step_{payload['task_id']}", step)
    
    elif etype == "TASK_STARTED":
        step = cl.user_session.get(f"step_{payload['task_id']}")
        if step:
            step.name = "Executing Task"
            await step.update()
    
    elif etype == "TASK_COMPLETED":
        step = cl.user_session.get(f"step_{payload['task_id']}")
        if step:
            model = payload.get("result", {}).get("model_used", "unknown-model")
            step.name = f"Task Completed ({model})"
            step.output = f"Result:\n```\n{payload['result'].get('stdout', '')}\n```"
            await step.update()

    elif etype == "TASK_FAILED":
        step = cl.user_session.get(f"step_{payload['task_id']}")
        if step:
            step.name = "Task Failed ❌"
            step.output = f"Error:\n```\n{payload.get('error', 'Unknown error')}\n```"
            await step.update()
            
    elif etype == "ESCALATION_STARTED":
        step = cl.user_session.get(f"step_{payload['task_id']}")
        if step:
            step.name = f"Escalating to {payload['next_model']} 🚀"
            step.output = f"Retrying task due to failure: {payload['error']}"
            await step.update()
            
    elif etype == "ARTIFACT_CREATED":
        if payload["type"] == "plot":
            image_bytes = base64.b64decode(payload["content_json"]["image_base64"])
            image = cl.Image(content=image_bytes, name="plot", display="inline")
            await cl.Message(content=f"🎨 **Visualization**: {payload['description']}", elements=[image]).send()
        else:
            await cl.Message(content=f"📝 **Artifact**: {payload['description']}").send()

    elif etype == "WORKFLOW_FINAL_RESULT":
        content = f"### 📖 The Story\n\n{payload['narrative']}\n\n"
        if payload.get("takeaways"):
            content += "**Key Takeaways:**\n"
            for t in payload["takeaways"]:
                content += f"- {t}\n"
        await cl.Message(content=content).send()

    elif etype == "STATE_UPDATED":
        files = payload.get("files", [])
        state = payload.get("state", {})
        
        # Format Unified Markdown
        md = "## 📁 Workspace Files\n"
        if not files:
            md += "*(No files yet)*\n"
        else:
            for f in files:
                md += f"- `{f}`\n"
        
        md += "\n---\n\n## 🧠 Sandbox Memory\n"
        if not state:
            md += "*(Empty)*\n"
        else:
            for var_name, var_info in state.items():
                vtype = var_info.get("type", "Unknown")
                md += f"- **`{var_name}`** (`{vtype}`)\n"
                if vtype == "DataFrame":
                    md += f"  - Shape: {var_info.get('shape')}\n"
                    cols_str = ", ".join([str(c) for c in var_info.get('columns', [])])
                    md += f"  - Cols: [{cols_str}]\n"
                elif "value" in var_info:
                    val = str(var_info['value'])[:100].replace('\n', ' ')
                    md += f"  - Value: `{val}`\n"
        
        explorer_el = cl.user_session.get("explorer_el")
        if explorer_el:
            explorer_el.content = md
            await explorer_el.update()

    elif etype == "STEP_COMPLETED":
        await cl.Message(content=f"✅ {payload['message']}").send()

@cl.on_message
async def main(message: cl.Message):
    files = []
    for element in message.elements:
        if isinstance(element, cl.File):
            content = element.content
            if content is None and element.path:
                with open(element.path, "rb") as f:
                    content = f.read()
            
            if content:
                files.append({
                    "name": element.name,
                    "content_base64": base64.b64encode(content).decode("utf-8")
                })
            
    project_id = cl.user_session.get("current_project_id")
    
    payload = {
        "name": "Chat Session Project",
        "objective": message.content,
        "files": files,
        "existing_project_id": str(project_id) if project_id else None
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(f"{BACKEND_URL}/projects", json=payload)
            resp.raise_for_status()
            project_data = resp.json()
            if not project_id:
                cl.user_session.set("current_project_id", project_data["id"])
        except Exception as e:
            await cl.Message(content=f"Error communicating with backend: {e}").send()
