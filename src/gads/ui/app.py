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

def get_action_buttons():
    return [
        cl.Action(name="upload_data", label="📁 Upload to Workspace", payload={"action": "upload"}),
        cl.Action(name="clear_session", label="🗑️ Reset Session", payload={"action": "clear"})
    ]

def _render_state_md(files, state) -> str:
    md = "### 📁 Workspace Files\n"
    if not files:
        md += "*(No files yet)*\n"
    else:
        for f in files:
            md += f"- `{f}`\n"

    md += "\n---\n\n### 🧠 Sandbox Memory\n"
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
    return md

async def sync_dashboard(files, state):
    """Refresh the persistent side panel via Chainlit's ElementSidebar API."""
    md = _render_state_md(files, state)
    state_el = cl.Text(name="ProjectState", content=md)
    await cl.ElementSidebar.set_elements([state_el])

@cl.on_chat_start
async def start():
    # Clear any previous session state to prevent bleeding
    cl.user_session.set("last_seq", 0)
    cl.user_session.set("current_project_id", None)
    
    # 1. Send the welcome message with actions
    dashboard_msg = cl.Message(
        content="--- 🚀 GADS: Data Science Rockstar Control Center ---\n\n"
                "Environment Initialized. Track workspace state in the side panel →",
        actions=get_action_buttons()
    )
    await dashboard_msg.send()

    # 2. Initialize the persistent side panel
    await cl.ElementSidebar.set_title("Project State")
    await sync_dashboard([], {})

    if not cl.user_session.get("ws_active"):
        cl.user_session.set("ws_active", True)
        asyncio.create_task(ws_listener())

@cl.action_callback("upload_data")
async def on_upload_action(action: cl.Action):
    try:
        files = await cl.AskFileMessage(
            content="Select files to upload to your workspace:",
            accept=["text/plain", "text/csv", "application/json", "application/vnd.ms-excel"],
            max_files=10,
            timeout=180,
            raise_on_timeout=False
        ).send()
        
        if files:
            upload_payload = []
            for f in files:
                with open(f.path, "rb") as fd:
                    content = fd.read()
                upload_payload.append({
                    "name": f.name,
                    "content_base64": base64.b64encode(content).decode("utf-8")
                })
                
            project_id = cl.user_session.get("current_project_id")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    if project_id:
                        url = f"{BACKEND_URL}/projects/{project_id}/files"
                        resp = await client.post(url, json={"files": upload_payload})
                    else:
                        payload = {"name": "Upload Session", "objective": "", "files": upload_payload}
                        resp = await client.post(f"{BACKEND_URL}/projects", json=payload)
                    
                    resp.raise_for_status()
                    data = resp.json()
                    
                    if not project_id:
                        cl.user_session.set("current_project_id", data["project"]["id"])
                    
                    # Update Sidebar
                    await sync_dashboard(data.get("files", []), {})
                    await cl.Message(content=f"✅ Successfully uploaded {len(files)} file(s).").send()
                    
                except Exception as e:
                    await cl.Message(content=f"❌ Upload failed: {e}").send()
    finally:
        await action.remove()
        # AskFileMessage.send() ends with an internal task_start() intended to
        # resume the enclosing message handler. Action callbacks aren't
        # task-wrapped, so without a matching task_end() the input bar stays
        # locked in "running" state.
        try:
            await cl.context.emitter.task_end()
        except Exception:
            pass

@cl.action_callback("clear_session")
async def on_clear_action(action: cl.Action):
    try:
        cl.user_session.set("current_project_id", None)
        await cl.Message(content="🔄 Session reset.").send()
    finally:
        await action.remove()

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
        except Exception:
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
            
    elif etype == "ARTIFACT_CREATED":
        if payload["type"] == "plot":
            image_bytes = base64.b64decode(payload["content_json"]["image_base64"])
            image = cl.Image(content=image_bytes, name="plot", display="inline")
            await cl.Message(content=f"🎨 Visualization: {payload['description']}", elements=[image]).send()

    elif etype == "WORKFLOW_FINAL_RESULT":
        content = f"### 📖 The Story\n\n{payload['narrative']}\n\n"
        if payload.get("takeaways"):
            content += "**Key Takeaways:**\n"
            for t in payload["takeaways"]:
                content += f"- {t}\n"
        await cl.Message(content=content).send()

    elif etype == "STATE_UPDATED":
        await sync_dashboard(payload.get("files", []), payload.get("state", {}))

    elif etype == "STEP_COMPLETED":
        await cl.Message(content=f"✅ {payload['message']}").send()

@cl.on_message
async def main(message: cl.Message):
    # Immediate ack to unlock UI if it's just an upload
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
        "name": "Chat Session",
        "objective": message.content,
        "files": files,
        "existing_project_id": str(project_id) if project_id else None
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            if files and not message.content.strip() and project_id:
                resp = await client.post(f"{BACKEND_URL}/projects/{project_id}/files", json={"files": files})
            else:
                resp = await client.post(f"{BACKEND_URL}/projects", json=payload)
            
            resp.raise_for_status()
            data = resp.json()
            
            if not project_id:
                cl.user_session.set("current_project_id", data["project"]["id"])
            
            # Sync dashboard
            await sync_dashboard(data.get("files", []), {})
            
            if files and not message.content.strip():
                await cl.Message(content=f"✅ Uploaded {len(files)} file(s).").send()
                
        except Exception as e:
            await cl.Message(content=f"Error: {e}").send()
