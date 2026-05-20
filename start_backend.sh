#!/bin/bash
export PYTHONPATH=src
./.venv/bin/python -m uvicorn gads.core.server:app --host 127.0.0.1 --port 8001
