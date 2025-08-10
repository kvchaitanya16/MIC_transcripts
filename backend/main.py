import os
import re
import io
import wave
import math
import difflib
import asyncio
import tempfile
import subprocess
import logging
import array
from typing import Dict, Optional, Tuple, Any, List
from contextlib import suppress as contextlib_suppress

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState
import uvicorn

from openai import OpenAI

# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ------------------ Config ------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")  # or "whisper-1"
STT_RESPONSE_FORMAT = os.getenv("STT_RESPONSE_FORMAT", "json").lower()      # "json" | "verbose_json"
PARTIAL_INTERVAL_SEC = float(os.getenv("PARTIAL_INTERVAL_SEC", "0.6"))

# PCM settings
PCM_SAMPLE_RATE = int(os.getenv("PCM_SAMPLE_RATE", "16000"))
PCM_CHANNELS = 1
PCM_SAMPWIDTH = 2  # 16-bit

# batching
MIN_BATCH_SEC   = float(os.getenv("MIN_BATCH_SEC", "1.2"))   # wait for >= 1.2s of NEW PCM
OVERLAP_SEC     = float(os.getenv("OVERLAP_SEC", "0.25"))
MAX_BATCH_SEC   = float(os.getenv("MAX_BATCH_SEC", "5.0"))
MAX_BUFFER_SEC  = float(os.getenv("MAX_BUFFER_SEC", "120"))

# VAD / Silence gate
AUDIO_LANGUAGE  = os.getenv("AUDIO_LANGUAGE", "auto")        # "auto" | "en" | "te" | etc.
MIN_RMS_ABS     = float(os.getenv("MIN_RMS_ABS", "250"))     # absolute floor for RMS
NOISE_FACTOR    = float(os.getenv("NOISE_FACTOR", "3.0"))    # threshold = max(MIN_RMS_ABS, noise_floor * NOISE_FACTOR)
NOISE_WINSIZE   = int(os.getenv("NOISE_WINSIZE", "12"))      # how many low-RMS windows to learn baseline
CLEAR_DRAFT_ON_SILENCE = os.getenv("CLEAR_DRAFT_ON_SILENCE", "1") == "1"

# Derived
BYTES_PER_SEC    = PCM_SAMPLE_RATE * PCM_CHANNELS * PCM_SAMPWIDTH
MIN_BATCH_BYTES  = int(MIN_BATCH_SEC * BYTES_PER_SEC)
OVERLAP_BYTES    = int(OVERLAP_SEC * BYTES_PER_SEC)
MAX_BATCH_BYTES  = int(MAX_BATCH_SEC * BYTES_PER_SEC)
MAX_BUFFER_BYTES = int(MAX_BUFFER_SEC * BYTES_PER_SEC)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------ Text helpers ------------------
SENT_ENDERS = set([".", "?", "!", "…", "।", "॥"])

def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def is_similar(a: str, b: str, thr: float = 0.90) -> bool:
    if not a or not b:
        return False
    return difflib.SequenceMatcher(a=normalize_text(a), b=normalize_text(b)).ratio() >= thr

def tokenize_sentences(text: str) -> Tuple[List[str], str]:
    out, buf = [], []
    for ch in text:
        buf.append(ch)
        if ch in SENT_ENDERS:
            sent = normalize_text("".join(buf))
            if sent:
                out.append(sent)
            buf = []
    draft = normalize_text("".join(buf)) if buf else ""
    return out, draft

def get_field(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

# ------------------ Audio utils ------------------
def compute_rms_s16le(pcm_bytes: bytes) -> float:
    """RMS of int16 little-endian PCM."""
    if not pcm_bytes:
        return 0.0
    arr = array.array("h")  # signed short
    arr.frombytes(pcm_bytes)
    if len(arr) == 0:
        return 0.0
    # avoid overflow: use float accumulation
    s = 0.0
    for v in arr:
        s += float(v) * float(v)
    return math.sqrt(s / len(arr))

# ------------------ Session state ------------------
class SessionState:
    def __init__(self):
        # Rolling PCM buffer (decoded by a persistent ffmpeg)
        self.pcm = bytearray()
        self.processed_bytes = 0
        self.pcm_lock = asyncio.Lock()

        # Persistent ffmpeg process + readers
        self.ffmpeg_proc: Optional[asyncio.subprocess.Process] = None
        self.ffmpeg_reader_task: Optional[asyncio.Task] = None
        self.ffmpeg_stderr_task: Optional[asyncio.Task] = None

        # Live transcription assembly
        self.confirmed_text = ""
        self.draft_text = ""
        self.seen_norm_sentences: set[str] = set()
        self.last_confirmed_norm: str = ""

        # VAD state
        self.noise_samples: List[float] = []  # low-RMS windows to estimate noise floor
        self.noise_floor: Optional[float] = None

        self.closed = False
        self.transcribe_task: Optional[asyncio.Task] = None

    @property
    def cum_text(self) -> str:
        if self.draft_text:
            need_space = bool(self.confirmed_text) and not self.confirmed_text.endswith((" ", "\n"))
            return (self.confirmed_text + (" " if need_space else "") + self.draft_text).strip()
        return self.confirmed_text

# in-memory sessions
sessions: Dict[WebSocket, SessionState] = {}

async def safe_send_json(ws: WebSocket, payload: dict):
    try:
        if ws.application_state == WebSocketState.CONNECTED:
            await ws.send_json(payload)
    except Exception:
        pass

# ------------------ ffmpeg: persistent transcode ------------------
async def start_ffmpeg(state: SessionState):
    """
    Launch one ffmpeg per websocket.
    We stream media chunks to stdin, and read decoded PCM from stdout.
    Let ffmpeg auto-detect the container (webm/ogg/mp4).
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-i", "pipe:0",             # detect from first chunk header
            "-ac", str(PCM_CHANNELS),
            "-ar", str(PCM_SAMPLE_RATE),
            "-f", "s16le",
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        raise RuntimeError("ffmpeg is not installed or not on PATH.")

    state.ffmpeg_proc = proc

    async def read_stdout():
        try:
            while True:
                chunk = await proc.stdout.read(4096)
                if not chunk:
                    break
                async with state.pcm_lock:
                    state.pcm.extend(chunk)
        except Exception as e:
            logging.error("ffmpeg stdout reader error: %s", e)

    async def read_stderr():
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                logging.debug("ffmpeg: %s", line.decode("utf-8", errors="ignore").strip())
        except Exception:
            pass

    state.ffmpeg_reader_task = asyncio.create_task(read_stdout())
    state.ffmpeg_stderr_task = asyncio.create_task(read_stderr())

async def stop_ffmpeg(state: SessionState):
    proc = state.ffmpeg_proc
    if not proc:
        return
    with contextlib_suppress(Exception):
        if proc.stdin and not proc.stdin.at_eof():
            proc.stdin.write_eof()
    with contextlib_suppress(asyncio.TimeoutError):
        await asyncio.wait_for(proc.wait(), timeout=2.0)

    for t in (state.ffmpeg_reader_task, state.ffmpeg_stderr_task):
        if t:
            t.cancel()
    state.ffmpeg_proc = None
    state.ffmpeg_reader_task = None
    state.ffmpeg_stderr_task = None

# ------------------ STT + merge ------------------
def transcribe_wav_bytes(wav_bytes: bytes, fmt: str) -> Any:
    with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmp:
        tmp.write(wav_bytes)
        tmp.flush()
        params = dict(model=TRANSCRIBE_MODEL, file=open(tmp.name, "rb"), response_format=fmt, temperature=0)
        # Optional language lock
        if AUDIO_LANGUAGE and AUDIO_LANGUAGE.lower() != "auto":
            params["language"] = AUDIO_LANGUAGE
        try:
            return client.audio.transcriptions.create(**params)
        finally:
            try:
                params["file"].close()
            except Exception:
                pass

def merge_json_text(state: SessionState, full_text: str) -> bool:
    changed = False
    full_text = normalize_text(full_text)
    if not full_text:
        return False

    sents, draft = tokenize_sentences(full_text)

    # integrate completed sentences
    for s in sents:
        norm = normalize_text(s)
        if norm in state.seen_norm_sentences:
            continue
        if state.last_confirmed_norm and is_similar(norm, state.last_confirmed_norm, thr=0.92):
            continue
        need_space = bool(state.confirmed_text) and not state.confirmed_text.endswith((" ", "\n"))
        state.confirmed_text = (state.confirmed_text + (" " if need_space else "") + s).strip()
        state.seen_norm_sentences.add(norm)
        state.last_confirmed_norm = norm
        changed = True

    # update draft
    if draft:
        if is_similar(draft, state.draft_text, thr=0.90):
            if len(draft) > len(state.draft_text):
                state.draft_text = draft
                changed = True
        else:
            state.draft_text = draft
            changed = True
    else:
        if state.draft_text:
            state.draft_text = ""
            changed = True

    return changed

def merge_response_into_state(state: SessionState, resp: Any, fmt: str) -> bool:
    if fmt == "verbose_json":
        txt = (get_field(resp, "text") or "").strip()
        if txt:
            return merge_json_text(state, txt)
        segs = get_field(resp, "segments") or []
        if segs:
            txt_joined = normalize_text(" ".join([(seg.get("text") or "").strip() for seg in segs]))
            return merge_json_text(state, txt_joined)
        return False
    else:
        txt = (get_field(resp, "text") or "").strip()
        if not txt:
            return False
        return merge_json_text(state, txt)

# ------------------ Partial loop (with VAD) ------------------
async def run_partial_loop(ws: WebSocket, state: SessionState):
    try:
        while not state.closed and ws.application_state == WebSocketState.CONNECTED:
            await asyncio.sleep(PARTIAL_INTERVAL_SEC)

            async with state.pcm_lock:
                total = len(state.pcm)
                new_bytes = total - state.processed_bytes
                if new_bytes < MIN_BATCH_BYTES:
                    continue

                start = max(0, state.processed_bytes - OVERLAP_BYTES)
                end = min(total, state.processed_bytes + min(new_bytes, MAX_BATCH_BYTES))
                slice_pcm = bytes(state.pcm[start:end])

            # ---- VAD: skip silence/low energy ----
            rms = compute_rms_s16le(slice_pcm)

            # learn noise floor (use only low-energy slices)
            if (state.noise_floor is None or rms <= (state.noise_floor * NOISE_FACTOR)):
                if len(state.noise_samples) < NOISE_WINSIZE:
                    state.noise_samples.append(rms)
                    # robust baseline: median of collected samples
                    sorted_ns = sorted(state.noise_samples)
                    state.noise_floor = sorted_ns[len(sorted_ns)//2] if sorted_ns else rms

            threshold = max(MIN_RMS_ABS, (state.noise_floor or 0.0) * NOISE_FACTOR)

            if rms < threshold:
                # Treat as silence → advance pointer but don't call STT
                if CLEAR_DRAFT_ON_SILENCE and state.draft_text:
                    state.draft_text = ""  # clear dangling draft during silence
                    await safe_send_json(ws, {"type": "partial", "text": state.cum_text})
                async with state.pcm_lock:
                    state.processed_bytes = end - OVERLAP_BYTES
                    if state.processed_bytes < 0:
                        state.processed_bytes = 0
                    if state.processed_bytes > MAX_BUFFER_BYTES:
                        drop = state.processed_bytes - MAX_BUFFER_BYTES
                        state.pcm = bytearray(state.pcm[drop:])
                        state.processed_bytes -= drop
                logging.debug(f"Silence gated (RMS={rms:.1f}, thr={threshold:.1f})")
                continue

            # ---- Transcribe speech slice ----
            with io.BytesIO() as wav_io:
                with wave.open(wav_io, "wb") as wf:
                    wf.setnchannels(PCM_CHANNELS)
                    wf.setsampwidth(PCM_SAMPWIDTH)
                    wf.setframerate(PCM_SAMPLE_RATE)
                    wf.writeframes(slice_pcm)
                wav_bytes = wav_io.getvalue()

            try:
                resp = transcribe_wav_bytes(wav_bytes, STT_RESPONSE_FORMAT)
                if merge_response_into_state(state, resp, STT_RESPONSE_FORMAT):
                    await safe_send_json(ws, {"type": "partial", "text": state.cum_text})
            except Exception as e:
                logging.error("Transcribe failed: %s", e)
                await safe_send_json(ws, {"type": "warning", "message": f"Transcribe failed: {e}"})

            # advance pointer & bound memory
            async with state.pcm_lock:
                state.processed_bytes = end - OVERLAP_BYTES
                if state.processed_bytes < 0:
                    state.processed_bytes = 0
                if state.processed_bytes > MAX_BUFFER_BYTES:
                    drop = state.processed_bytes - MAX_BUFFER_BYTES
                    state.pcm = bytearray(state.pcm[drop:])
                    state.processed_bytes -= drop
                logging.info("PCM total=%dB processed=%dB new=%dB (RMS=%.1f thr=%.1f)",
                             len(state.pcm), state.processed_bytes, len(state.pcm) - state.processed_bytes, rms, threshold)

    except asyncio.CancelledError:
        pass

# ------------------ WebSocket ------------------
@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket):
    await ws.accept()
    state = SessionState()
    sessions[ws] = state

    # Start ffmpeg and partial loop
    await start_ffmpeg(state)
    state.transcribe_task = asyncio.create_task(run_partial_loop(ws, state))
    await safe_send_json(ws, {"type": "ready", "message": "Mic stream accepted. Send binary audio chunks."})

    try:
        while True:
            msg = await ws.receive()
            if "type" in msg and msg["type"] == "websocket.disconnect":
                break

            if msg.get("text") is not None:
                continue

            data = msg.get("bytes")
            if data:
                try:
                    # Stream raw media chunk straight into ffmpeg stdin
                    if state.ffmpeg_proc and state.ffmpeg_proc.stdin:
                        state.ffmpeg_proc.stdin.write(data)
                        await state.ffmpeg_proc.stdin.drain()
                    else:
                        logging.error("ffmpeg process not available while receiving data")
                    await safe_send_json(ws, {"type": "ack", "bytes": len(data)})
                except Exception as e:
                    logging.error("Audio stream write failed: %s", e)
                    await safe_send_json(ws, {"type": "warning", "message": f"Audio stream write failed: {e}"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logging.exception("WebSocket error: %s", e)
        await safe_send_json(ws, {"type": "error", "message": str(e)})
    finally:
        state.closed = True
        if state.transcribe_task:
            state.transcribe_task.cancel()
            with contextlib_suppress(asyncio.CancelledError):
                await state.transcribe_task

        # final pass on any remaining PCM (if not silence)
        try:
            async with state.pcm_lock:
                total = len(state.pcm)
                new_bytes = total - state.processed_bytes
                if new_bytes > 0 and ws.application_state == WebSocketState.CONNECTED:
                    start = max(0, state.processed_bytes - OVERLAP_BYTES)
                    end = total
                    slice_pcm = bytes(state.pcm[start:end])

            if new_bytes > 0 and ws.application_state == WebSocketState.CONNECTED:
                rms = compute_rms_s16le(slice_pcm)
                thr = max(MIN_RMS_ABS, (state.noise_floor or 0.0) * NOISE_FACTOR)
                if rms >= thr:
                    with io.BytesIO() as wav_io:
                        with wave.open(wav_io, "wb") as wf:
                            wf.setnchannels(PCM_CHANNELS)
                            wf.setsampwidth(PCM_SAMPWIDTH)
                            wf.setframerate(PCM_SAMPLE_RATE)
                            wf.writeframes(slice_pcm)
                        wav_bytes = wav_io.getvalue()
                    resp = transcribe_wav_bytes(wav_bytes, STT_RESPONSE_FORMAT)
                    merge_response_into_state(state, resp, STT_RESPONSE_FORMAT)
                await safe_send_json(ws, {"type": "final", "text": state.cum_text})
        except Exception as e:
            logging.error("Final transcribe failed: %s", e)

        await stop_ffmpeg(state)
        sessions.pop(ws, None)
        with contextlib_suppress(Exception):
            if ws.application_state == WebSocketState.CONNECTED:
                await ws.close()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
