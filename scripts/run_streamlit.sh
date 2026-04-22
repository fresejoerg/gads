#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
uv run streamlit run src/gads/ui/streamlit_app.py --server.port 8003
