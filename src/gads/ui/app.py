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
    await cl.Message(content="--- 🚀 GADS: Data Science Rockstar Control Center ---").send()
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
        # Display the story as a main message
        content = f"### 📖 The Story\n\n{payload['narrative']}\n\n"
        if payload.get("takeaways"):
            content += "**Key Takeaways:**\n"
            for t in payload["takeaways"]:
                content += f"- {t}\n"
        await cl.Message(content=content).send()

    elif etype == "STEP_COMPLETED":
        await cl.Message(content=f"✅ {payload['message']}").send()

@cl.on_message
async def main(message: cl.Message):
    async with httpx.AsyncClient() as client:
        try:
            await client.post(f"{BACKEND_URL}/projects", params={
                "name": "User Project",
                "objective": message.content
            })
        except Exception as e:
            await cl.Message(content=f"Error initializing project: {e}").send()
