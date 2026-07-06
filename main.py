#!/usr/bin/env python3
"""Railway entry point — reads $PORT and runs uvicorn."""
import os
import sys

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting on port {port} (from PORT env)", flush=True)
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, log_level="info")
