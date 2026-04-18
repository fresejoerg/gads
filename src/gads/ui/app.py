import os
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
    
    # Start background task to listen to WebSocket
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
        except (websockets.ConnectionClosed, ConnectionRefusedError, OSError) as e:
            print(f"WS Connection Error: {e}. Retrying in 2s...")
            await asyncio.sleep(2)
        except Exception as e:
            print(f"WS Listener Crash: {e}. Retrying in 5s...")
            await asyncio.sleep(5)

async def handle_event(event: dict):
    etype = event["type"]
    payload = event["payload"]
    
    if etype == "TASK_STARTED":
        async with cl.Step(name="Task Execution") as step:
            step.input = f"Starting task: {payload['task_id']}"
            cl.user_session.set(f"step_{payload['task_id']}", step)
    
    elif etype == "TASK_COMPLETED":
        step = cl.user_session.get(f"step_{payload['task_id']}")
        if step:
            step.output = f"Task completed successfully."
            await step.update()
            
    elif etype == "ARTIFACT_CREATED":
        await cl.Message(content=f"🎨 New Artifact: {payload['description']}").send()
        if payload["type"] == "plot":
            # Render plot
            image = cl.Image(content=payload["content_json"]["image_base64"], name="plot", display="inline")
            await cl.Message(content="", elements=[image]).send()

@cl.on_message
async def main(message: cl.Message):
    # This would call the backend to create a project/run
    async with httpx.AsyncClient() as client:
        try:
            await client.post(f"{BACKEND_URL}/projects", params={
                "name": "New Project",
                "objective": message.content
            })
            await cl.Message(content=f"Project initialized: '{message.content}'. Watching the agent DAG...").send()
        except Exception as e:
            await cl.Message(content=f"Error initializing project: {e}").send()
