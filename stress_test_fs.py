import json
import urllib.request
import os

def stress_test():
    prompt_path = '/home/jfrese/projects/GADS/stress_test_fs_prompt.txt'
    with open(prompt_path, 'r') as f:
        prompt = f.read()

    url = "http://localhost:4000/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-1234"
    }
    data = {
        "model": "claude-opus-4.7",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers)
    
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            print(res_data['choices'][0]['message']['content'])
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    stress_test()
