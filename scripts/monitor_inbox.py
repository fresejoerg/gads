import os
import json
import time
from datetime import datetime

INBOX_PATH = "/home/joergf/projects/GADS/agent_inbox.jsonl"

def get_timestamp():
    return datetime.now().isoformat()

def get_current_plan():
    plan_path = "/home/joergf/projects/GADS/agents_collab_NEXT.md"
    if os.path.exists(plan_path):
        try:
            with open(plan_path, "r") as f:
                content = f.read()
            if "## Proposed Work Plan" in content:
                plan_part = content.split("## Proposed Work Plan")[1].strip()
                if "## Antigravity's Review Notes" in plan_part:
                    plan_part = plan_part.split("## Antigravity's Review Notes")[0].strip()
                return plan_part
            return content[:1000]
        except Exception as e:
            return f"Error reading plan: {e}"
    return "No active plan file found."

def append_reply(to, msg_type, content):
    reply = {
        "from": "Antigravity",
        "to": to,
        "ts": get_timestamp(),
        "type": msg_type,
        "content": content
    }
    with open(INBOX_PATH, "a") as f:
        f.write(json.dumps(reply) + "\n")
        f.flush()

def main():
    print("Starting Antigravity agent inbox monitor...", flush=True)
    
    # If file doesn't exist, wait or create it
    if not os.path.exists(INBOX_PATH):
        with open(INBOX_PATH, "w") as f:
            pass
            
    # Open file and seek to end so we only read new lines
    with open(INBOX_PATH, "r") as f:
        f.seek(0, os.SEEK_END)
        
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.5)
                continue
                
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
            except Exception as e:
                print(f"Failed to parse line: {line}. Error: {e}", flush=True)
                continue
                
            # Filter messages
            to_field = data.get("to")
            from_field = data.get("from")
            
            # Avoid self messages or recursive replies
            if from_field == "Antigravity":
                continue
                
            if to_field in ("all", "Antigravity"):
                print(f"[INBOX_MATCH] Received message: {line}", flush=True)
                
                # Append ack back
                sender = from_field or "user"
                task_content = data.get("content", "")
                
                # Send ack with current plan
                plan_str = get_current_plan()
                ack_content = f"Acknowledged task: {task_content}\n\n[Antigravity Current Plan]\n{plan_str}"
                append_reply(to=sender, msg_type="ack", content=ack_content)

if __name__ == "__main__":
    main()
