import os
import base64
from datetime import datetime

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
        cl.Action(name="upload_data", label="Upload to Workspace", payload={"action": "upload"}),
        cl.Action(name="clear_session", label="Reset Session", payload={"action": "clear"}),
        cl.Action(name="refresh_archive", label="Refresh Archive", payload={"action": "refresh"})
    ]

async def sync_archive():
    """Fetch all past projects and display them in a persistent message."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{BACKEND_URL}/projects")
            resp.raise_for_status()
            projects = resp.json()

        if not projects:
            md = "### Project Archive\n*(No past projects found)*"
        else:
            md = "### Project Archive\n"
            # Show last 10 projects for context efficiency
            for p in projects[:10]:
                p_id = p["id"]
                p_name = p["name"] or "Unnamed"
                md += f"- **{p_name}** (`{p_id[:8]}`) [view dashboard]({BACKEND_URL}/projects/{p_id}/files/final_dashboard.html)\n"
            
            if len(projects) > 10:
                md += f"\n*(Showing 10 of {len(projects)} total projects)*"

        await cl.Message(content=md, author="System Archive").send()
    except Exception as e:
        print(f"[UI] Error syncing archive: {e}")

def _render_state_md(files, state, project_id) -> str:
    md = "### Workspace Files\n"
    if not files:
        md += "*(No files yet)*\n"
    else:
        for f in files:
            is_img = f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))
            icon = "[IMAGE]" if is_img else "[FILE]"
            # Direct link using BACKEND_URL and project_id as requested
            md += f"- {icon} `{f}` ([view]({BACKEND_URL}/projects/{project_id}/files/{f}) | [download]({BACKEND_URL}/projects/{project_id}/files/{f}?download=true))\n"

    md += "\n---\n\n### Sandbox Memory\n"
    if not state:
        md += "*(Empty)*\n"
    else:
        # Sort and truncate to prevent UI lag if variables exceed 50
        sorted_vars = sorted(state.items())
        display_vars = sorted_vars[:50]
        for var_name, var_info in display_vars:
            vtype = var_info.get("type", "Unknown")
            md += f"- **`{var_name}`** (`{vtype}`)\n"
            if vtype == "DataFrame":
                md += f"  - Shape: {var_info.get('shape')}\n"
            elif "value" in var_info:
                val = str(var_info['value'])[:100].replace('\n', ' ')
                md += f"  - Value: `{val}`\n"
        
        if len(sorted_vars) > 50:
            md += f"\n*(Truncated: {len(sorted_vars) - 50} more variables...)*\n"
            
    return md

async def sync_dashboard(files, state):
    """Refresh the persistent side panel via Chainlit's ElementSidebar API."""
    project_id = cl.user_session.get("current_project_id")
    if not project_id:
        print(f"[UI] {datetime.now().strftime('[%H:%M:%S]')} Warning: Sync attempted but no current_project_id found.")
        return

    try:
        md = _render_state_md(files, state, project_id)
        state_el = cl.Text(name="ProjectState", content=md)
        
        # Update elements
        await cl.ElementSidebar.set_elements([state_el])
        
        # Persistence check: Only force open if we haven't successfully synced in this session
        # or if the user explicitly reset. This prevents rapid "re-opening" jitter.
        if not cl.user_session.get("sidebar_initialized"):
            await cl.ElementSidebar.open()
            cl.user_session.set("sidebar_initialized", True)
            
    except Exception as e:
        print(f"[UI] {datetime.now().strftime('[%H:%M:%S]')} Error syncing dashboard: {e}")

@cl.on_chat_start
async def start():
    # Clear any previous session state to prevent bleeding
    cl.user_session.set("last_seq", 0)
    cl.user_session.set("current_project_id", None)
    cl.user_session.set("sidebar_initialized", False)
    
    print(f"[UI] {datetime.now().strftime('[%H:%M:%S]')} New chat session started.")
    
    # 1. Send the welcome message with actions
    now = datetime.now().strftime("[%H:%M:%S]")
    dashboard_msg = cl.Message(
        content=f"{now} --- GADS: Data Science Control Center ---\n\n"
                "Environment Initialized. Track workspace state and artifacts in the side panel.",
        actions=get_action_buttons()
    )
    await dashboard_msg.send()

    # 2. Sync and show the Project Archive
    await sync_archive()

    # 3. Initialize the persistent side panel
    await cl.ElementSidebar.set_title("Project State")
    
    if not cl.user_session.get("ws_active"):
        cl.user_session.set("ws_active", True)
        asyncio.create_task(ws_listener())

@cl.action_callback("refresh_archive")
async def on_refresh_archive(action: cl.Action):
    try:
        await sync_archive()
    finally:
        await action.remove()

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
                        project_id = data["project"]["id"]
                    
                    await sync_dashboard(data.get("files", []), {})
                    now = datetime.now().strftime("[%H:%M:%S]")
                    await cl.Message(content=f"{now} [SUCCESS] Successfully uploaded {len(files)} file(s).").send()
                    
                except Exception as e:
                    now = datetime.now().strftime("[%H:%M:%S]")
                    await cl.Message(content=f"{now} [FAILURE] Upload failed: {e}").send()
    finally:
        await action.remove()
        # Enforce task end to unlock the chat input
        try:
            await cl.context.emitter.task_end()
        except Exception:
            pass

@cl.action_callback("clear_session")
async def on_clear_action(action: cl.Action):
    try:
        cl.user_session.set("current_project_id", None)
        now = datetime.now().strftime("[%H:%M:%S]")
        await cl.Message(content=f"{now} 🔄 Session reset.").send()
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
    now = datetime.now().strftime("[%H:%M:%S]")
    
    if etype == "STEP_STARTED":
        await cl.Message(content=f"{now} [SYSTEM] {payload['message']}").send()

    elif etype == "TASK_CREATED":
        step = cl.Step(name="Task Planned", type="tool")
        step.input = f"{now} {payload['description']}"
        await step.send()
        cl.user_session.set(f"step_{payload['task_id']}", step)
    
    elif etype == "TASK_STARTED":
        step = cl.user_session.get(f"step_{payload['task_id']}")
        if step:
            step.name = f"{now} Executing Task"
            await step.update()
    
    elif etype == "TASK_COMPLETED":
        step = cl.user_session.get(f"step_{payload['task_id']}")
        if step:
            model = payload.get("result", {}).get("model_used", "unknown-model")
            step.name = f"{now} Task Completed ({model})"
            step.output = f"Result:\n```\n{payload['result'].get('stdout', '')}\n```"
            await step.update()

    elif etype == "TASK_FAILED":
        step = cl.user_session.get(f"step_{payload['task_id']}")
        if step:
            step.name = f"{now} Task Failed"
            step.output = f"Error:\n```\n{payload.get('error', 'Unknown error')}\n```"
            await step.update()
            
    elif etype == "ARTIFACT_CREATED":
        if payload["type"] == "plot":
            image_bytes = base64.b64decode(payload["content_json"]["image_base64"])
            image = cl.Image(content=image_bytes, name="plot", display="inline")
            await cl.Message(content=f"{now} [VISUALIZATION] {payload['description']}", elements=[image]).send()
        
        elif payload["type"] == "interactive_plot":
            project_id = payload.get("project_id")
            filename = payload["content_json"].get("filename")
            url = f"{BACKEND_URL}/projects/{project_id}/files/{filename}"
            await cl.Message(content=f"{now} [VISUALIZATION] {payload['description']} ([view interactive]({url}))").send()

    elif etype == "WORKFLOW_FINAL_RESULT":
        content = f"{now} ### Project Narrative\n\n{payload['narrative']}\n\n"
        if payload.get("takeaways"):
            content += "**Key Takeaways:**\n"
            for t in payload["takeaways"]:
                content += f"- {t}\n"
        await cl.Message(content=content).send()

    elif etype == "STATE_UPDATED":
        await sync_dashboard(payload.get("files", []), payload.get("state", {}))

    elif etype == "STEP_COMPLETED":
        await cl.Message(content=f"{now} [COMPLETE] {payload['message']}").send()

@cl.on_message
async def main(message: cl.Message):
    # Parse attached files
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
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            if files and not message.content.strip() and project_id:
                resp = await client.post(f"{BACKEND_URL}/projects/{project_id}/files", json={"files": files})
            else:
                payload = {
                    "name": "Chat Session",
                    "objective": message.content,
                    "files": files,
                    "existing_project_id": str(project_id) if project_id else None
                }
                resp = await client.post(f"{BACKEND_URL}/projects", json=payload)
            
            resp.raise_for_status()
            data = resp.json()
            
            if not project_id:
                cl.user_session.set("current_project_id", data["project"]["id"])
                project_id = data["project"]["id"]
            
            await sync_dashboard(data.get("files", []), {})
            
            if files and not message.content.strip():
                now = datetime.now().strftime("[%H:%M:%S]")
                await cl.Message(content=f"{now} [SUCCESS] Uploaded {len(files)} file(s).").send()
                
        except Exception as e:
            now = datetime.now().strftime("[%H:%M:%S]")
            await cl.Message(content=f"{now} Error: {e}").send()
