import os
import io
import asyncio
import tempfile
from typing import Dict, Optional
from contextlib import suppress as contextlib_suppress
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState
import uvicorn

# OpenAI Python SDK >= 1.0
from openai import OpenAI

# ---- config ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "whisper-1")  # e.g. "whisper-1" or "gpt-4o-mini-transcribe"
PARTIAL_INTERVAL_SEC = float(os.getenv("PARTIAL_INTERVAL_SEC", "1.0"))  # how often to push partials

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=OPENAI_API_KEY)

class SessionState:
    def __init__(self):
        # keep appending raw .webm chunks here
        self.buffer = io.BytesIO()
        self.last_bytes_len = 0
        self.closed = False
        self.transcribe_task: Optional[asyncio.Task] = None

# in-memory session store (1 per websocket)
sessions: Dict[WebSocket, SessionState] = {}

async def run_partial_loop(ws: WebSocket, state: SessionState):
    """Every PARTIAL_INTERVAL_SEC, run STT on current buffer and send partial text."""
    try:
        while not state.closed and ws.application_state == WebSocketState.CONNECTED:
            await asyncio.sleep(PARTIAL_INTERVAL_SEC)

            # if buffer grew since last run, transcribe again
            current_len = state.buffer.getbuffer().nbytes
            if current_len <= 0 or current_len == state.last_bytes_len:
                continue
            state.last_bytes_len = current_len

            # write temp .webm and transcribe
            state.buffer.seek(0)
            with tempfile.NamedTemporaryFile(delete=True, suffix=".webm") as tmp:
                tmp.write(state.buffer.read())
                tmp.flush()
                state.buffer.seek(0)

                try:
                    # New SDK (no open()) — pass a file-like object if desired.
                    # Here we re-open for clarity.
                    with open(tmp.name, "rb") as f:
                        resp = client.audio.transcriptions.create(
                            model=TRANSCRIBE_MODEL,
                            file=f,
                            # choose your format; "verbose_json" can include words/timestamps if the model supports it
                            response_format="json"
                        )
                    text = getattr(resp, "text", "") or (resp.get("text") if isinstance(resp, dict) else "")
                    if text:
                        await ws.send_json({"type": "partial", "text": text})
                except Exception as e:
                    # Send a non-fatal warning to the client; keep loop alive
                    await safe_send_json(ws, {"type": "warning", "message": f"Transcribe failed: {e}"})
    except asyncio.CancelledError:
        # graceful stop
        pass

async def safe_send_json(ws: WebSocket, payload: dict):
    try:
        if ws.application_state == WebSocketState.CONNECTED:
            await ws.send_json(payload)
    except Exception:
        pass

@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket):
    await ws.accept()
    state = SessionState()
    sessions[ws] = state
    state.transcribe_task = asyncio.create_task(run_partial_loop(ws, state))
    await safe_send_json(ws, {"type": "ready", "message": "Mic stream accepted. Send binary audio/webm chunks."})

    try:
        while True:
            msg = await ws.receive()
            if "type" in msg and msg["type"] == "websocket.disconnect":
                break

            if msg.get("text") is not None:
                continue

            data = msg.get("bytes")
            if data:
                state.buffer.seek(0, io.SEEK_END)
                state.buffer.write(data)
                await safe_send_json(ws, {"type": "ack", "bytes": len(data)})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await safe_send_json(ws, {"type": "error", "message": str(e)})
    finally:
        state.closed = True
        if state.transcribe_task:
            state.transcribe_task.cancel()
            with contextlib_suppress(asyncio.CancelledError):
                await state.transcribe_task

        # Try one final transcription
        try:
            if state.buffer.getbuffer().nbytes > 0 and ws.application_state == WebSocketState.CONNECTED:
                state.buffer.seek(0)
                with tempfile.NamedTemporaryFile(delete=True, suffix=".webm") as tmp:
                    tmp.write(state.buffer.read())
                    tmp.flush()
                    state.buffer.seek(0)
                    with open(tmp.name, "rb") as f:
                        resp = client.audio.transcriptions.create(
                            model=TRANSCRIBE_MODEL,
                            file=f,
                            response_format="json"
                        )
                    text = getattr(resp, "text", "") or (resp.get("text") if isinstance(resp, dict) else "")
                    if text:
                        await safe_send_json(ws, {"type": "final", "text": text})
        except Exception:
            pass

        sessions.pop(ws, None)

        # DO NOT blindly close; only if still connected
        with contextlib_suppress(Exception):
            if ws.application_state == WebSocketState.CONNECTED:
                await ws.close()

# tiny helper
from contextlib import suppress as contextlib_suppress

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
