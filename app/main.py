from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import shutil
import os
from .processing import analyze_audio

app = FastAPI(title="Adaptive Tone Engine")

@app.get("/")
async def root():
    return {"service": "Adaptive Tone Engine", "status": "prototype"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Accept an audio file upload and return analysis (placeholder).

    Expects standard audio MIME types (audio/wav, audio/mpeg, etc.).
    """
    if not (file.content_type and file.content_type.startswith("audio/")):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)
        result = analyze_audio(tmp_path)
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
    return JSONResponse(content=result)
