#!/usr/bin/env python3
"""Railway entry point — reads $PORT and runs uvicorn."""
import os
import sys

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Step 1: Starting main.py...", flush=True)
print(f"Step 2: PORT env = {os.environ.get('PORT', 'NOT SET')}", flush=True)
print(f"Step 3: Python = {sys.version}", flush=True)
print(f"Step 4: CWD = {os.getcwd()}", flush=True)
print(f"Step 5: Files in CWD = {os.listdir('.')}", flush=True)

# Try importing the app module early to catch errors
print("Step 6: Importing api_server...", flush=True)
try:
    import api_server
    print("Step 7: api_server imported successfully", flush=True)
except Exception as e:
    print(f"Step 7 FAILED: {e}", flush=True)
    import traceback
    traceback.print_exc(file=sys.stdout)
    sys.exit(1)

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Step 8: Starting uvicorn on port {port}", flush=True)
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, log_level="info", reload=False)
