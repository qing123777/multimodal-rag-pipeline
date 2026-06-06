import os
import json
import asyncio
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel

from RAG_pipeline import initialize_rag

app = FastAPI(title="CPS RAG Assistant")

# Initialized once at startup; blocks until ingestion is complete on first run
assistant = None


@app.on_event("startup")
def startup_event():
    global assistant
    print("Initializing RAG pipeline...")
    assistant = initialize_rag()
    print("RAG pipeline ready.")


# ── Models ────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


# ── Routes ───────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = Path(__file__).parent / "frontend.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/chat")
def chat(request: ChatRequest):
    """
    Stream the assistant's response as Server-Sent Events.
    Each event: data: {"chunk": "<text>"}\n\n
    Final event: data: [DONE]\n\n
    """
    # assistant.stream() is a blocking sync generator; run it in a background
    # thread and forward chunks to the async response via a queue.
    queue: "asyncio.Queue[str | None]" = asyncio.Queue()
    loop = asyncio.new_event_loop()

    def run_stream():
        try:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            for chunk in assistant.stream(request.message):
                # Use a thread-safe put to the main loop's queue
                main_loop.call_soon_threadsafe(queue.put_nowait, chunk)
        finally:
            main_loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    # We need the running event loop; use a sync generator approach instead
    def sync_generator():
        # Collect chunks from the blocking generator and yield SSE-formatted strings
        for chunk in assistant.stream(request.message):
            data = json.dumps({"chunk": chunk}, ensure_ascii=False)
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        sync_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
