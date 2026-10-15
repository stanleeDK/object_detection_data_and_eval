"""Start the web app:  uv run python -m app   ->  http://127.0.0.1:8000"""

import uvicorn

# 127.0.0.1 = only reachable from this Mac. Use --host 0.0.0.0 via uvicorn directly to share on your network.
uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
