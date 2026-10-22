import io
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soxr
import soundfile as sf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from needle.agent.whistle import Whistle, SAMPLE_RATE

app = FastAPI()
_whistle: Whistle | None = None


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)


def _get_whistle() -> Whistle:
    global _whistle
    if _whistle is None:
        print("Loading Whistle model…", flush=True)
        _whistle = Whistle()
        print("Whistle ready.", flush=True)
    return _whistle


def _to_mono_float32(data: bytes, filename: str) -> np.ndarray:
    suffix = Path(filename).suffix.lower()
    try:
        audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
    except Exception:
        # fallback: ffmpeg → raw f32le PCM
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            result = subprocess.run(
                ["ffmpeg", "-i", tmp_path, "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "f32le", "-", "-loglevel", "error"],
                capture_output=True, check=True,
            )
            return np.frombuffer(result.stdout, dtype=np.float32)
        finally:
            os.unlink(tmp_path)

    if rate != SAMPLE_RATE:
        audio = soxr.resample(audio, rate, SAMPLE_RATE, quality="HQ")
    return audio


@app.on_event("startup")
async def warmup():
    _get_whistle()


@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "transcribe.html").read_text()


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    data = await file.read()
    try:
        audio = _to_mono_float32(data, file.filename or "audio.wav")
        t0 = time.perf_counter()
        result = _get_whistle().transcribe(audio, language="en")
        elapsed_s = time.perf_counter() - t0
        audio_s = len(audio) / SAMPLE_RATE
        return {
            "text": result["text"],
            "audio_s": round(audio_s, 2),
            "elapsed_s": round(elapsed_s, 2),
            "rtf": round(elapsed_s / audio_s, 3) if audio_s else None,
            "ttft_ms": result["ttft_ms"],
            "decode_tps": round(result["decode_tps"], 1) if result["decode_tps"] else None,
        }
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
