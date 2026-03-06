"""
Option C — Lightweight backend using ONLY free APIs:
  - yt-dlp          (download videos, free)
  - FFmpeg          (cut clips, free)
  - Gemini API      (transcribe + describe, free tier)
  - No Whisper needed — Gemini handles audio directly!
"""

import os, uuid, json, subprocess, asyncio, base64
from pathlib import Path
import httpx
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="ClipForge — Free Tier")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path("uploads"); UPLOAD_DIR.mkdir(exist_ok=True)
CLIPS_DIR  = Path("clips");   CLIPS_DIR.mkdir(exist_ok=True)

app.mount("/clips", StaticFiles(directory="clips"), name="clips")

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

jobs: dict = {}

# ── MODELS ────────────────────────────────────────────────────
class LinkRequest(BaseModel):
    url: str
    num_clips: int = 5
    clip_length: int = 30
    focus: str = "engaging"

# ── ROUTES ────────────────────────────────────────────────────
@app.get("/")
def root(): return {"status": "ClipForge Free API running ✓"}

@app.post("/from-link")
async def from_link(req: LinkRequest, bg: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "progress": 0, "message": "Queued", "clips": [], "error": ""}
    bg.add_task(pipeline, job_id, req.url, req.num_clips, req.clip_length, req.focus)
    return {"job_id": job_id}

@app.get("/status/{job_id}")
def status(job_id: str):
    if job_id not in jobs: raise HTTPException(404, "Not found")
    return jobs[job_id]

@app.get("/download/{filename}")
def download(filename: str):
    p = CLIPS_DIR / filename
    if not p.exists(): raise HTTPException(404, "Clip not found")
    return FileResponse(p, media_type="video/mp4", filename=filename)

# ── PIPELINE ──────────────────────────────────────────────────
async def pipeline(job_id, url, num_clips, clip_length, focus):
    try:
        # STEP 1 — Download
        set_job(job_id, "downloading", 10, "Downloading video...")
        video_path = await download_video(job_id, url)

        # STEP 2 — Get duration
        set_job(job_id, "analyzing", 25, "Analyzing video...")
        duration = get_duration(video_path)
        timestamps = spread_timestamps(duration, num_clips, clip_length)

        # STEP 3 — Cut clips
        set_job(job_id, "clipping", 45, f"Cutting {len(timestamps)} clips...")
        clip_paths = cut_clips(job_id, video_path, timestamps, clip_length)

        # STEP 4 — Upload clips to Gemini Files API + Transcribe + Describe
        set_job(job_id, "transcribing", 65, "Transcribing with Gemini...")
        results = []
        for i, cp in enumerate(clip_paths):
            pct = 65 + int((i / len(clip_paths)) * 25)
            set_job(job_id, "writing", pct, f"Processing clip {i+1}/{len(clip_paths)}...")

            # Upload to Gemini Files API
            file_uri = await upload_to_gemini(cp)

            # Transcribe
            transcript = await gemini_transcribe(file_uri)

            # Generate description + metadata
            meta = await gemini_describe(transcript, focus)

            fname = Path(cp).name
            results.append({
                "filename": fname,
                "download_url": f"/download/{fname}",
                "stream_url": f"/clips/{fname}",
                "transcript": transcript,
                **meta
            })

        set_job(job_id, "done", 100, "Done!", clips=results)

    except Exception as e:
        jobs[job_id].update({"status": "error", "error": str(e), "progress": 0})
    finally:
        try: os.remove(video_path)
        except: pass


# ── HELPERS ───────────────────────────────────────────────────

def set_job(job_id, status, progress, message, clips=None):
    jobs[job_id].update({"status": status, "progress": progress, "message": message})
    if clips is not None:
        jobs[job_id]["clips"] = clips


async def download_video(job_id: str, url: str) -> str:
    out = str(UPLOAD_DIR / f"{job_id}.%(ext)s")
    cmd = [
        "yt-dlp",
        "--format", "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--output", out, "--no-playlist", url,
    ]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise Exception(f"Download failed: {err.decode()[:300]}")
    matches = list(UPLOAD_DIR.glob(f"{job_id}.*"))
    if not matches:
        raise Exception("Downloaded file not found after yt-dlp")
    return str(matches[0])


def get_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", path],
        capture_output=True, text=True
    )
    return float(json.loads(r.stdout).get("format", {}).get("duration", 0))


def spread_timestamps(duration: float, n: int, length: int) -> list:
    if duration <= length: return [0]
    usable = duration - length
    step = usable / max(n, 1)
    return [round(i * step) for i in range(n) if i * step + length <= duration]


def cut_clips(job_id: str, src: str, timestamps: list, length: int) -> list:
    out_files = []
    for i, start in enumerate(timestamps):
        out = str(CLIPS_DIR / f"{job_id}_clip{i+1}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start), "-i", src,
            "-t", str(length),
            # Auto-pad to 9:16 vertical (1080x1920)
            "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,"
                   "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", out
        ]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0:
            out_files.append(out)
    return out_files


async def upload_to_gemini(clip_path: str) -> str:
    """
    Upload video clip to Gemini Files API.
    Returns the file URI used in subsequent API calls.
    Gemini Files API is FREE — files expire after 48 hours.
    """
    file_size = os.path.getsize(clip_path)
    mime = "video/mp4"
    filename = Path(clip_path).name

    # 1. Initiate resumable upload
    async with httpx.AsyncClient(timeout=60) as client:
        init_resp = await client.post(
            f"{GEMINI_BASE}/files?key={GEMINI_KEY}",
            headers={
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(file_size),
                "X-Goog-Upload-Header-Content-Type": mime,
                "Content-Type": "application/json",
            },
            json={"file": {"display_name": filename}}
        )
        upload_url = init_resp.headers.get("x-goog-upload-url", "")
        if not upload_url:
            raise Exception("Failed to get Gemini upload URL")

        # 2. Upload file bytes
        with open(clip_path, "rb") as f:
            file_data = f.read()

        upload_resp = await client.post(
            upload_url,
            headers={
                "Content-Length": str(file_size),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            content=file_data,
        )
        result = upload_resp.json()
        return result.get("file", {}).get("uri", "")


async def gemini_transcribe(file_uri: str) -> str:
    """Ask Gemini to transcribe the video clip using its vision/audio capability."""
    if not file_uri:
        return ""

    prompt = "Listen to the audio in this video and transcribe exactly what is said. Return only the spoken words."

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{GEMINI_BASE}/models/gemini-2.0-flash:generateContent?key={GEMINI_KEY}",
            json={
                "contents": [{
                    "parts": [
                        {"file_data": {"mime_type": "video/mp4", "file_uri": file_uri}},
                        {"text": prompt}
                    ]
                }]
            }
        )
    try:
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return ""


async def gemini_describe(transcript: str, focus: str) -> dict:
    """Use Gemini to generate viral title, description, hashtags, and score."""
    focus_prompts = {
        "engaging": "most engaging and attention-grabbing",
        "funny":    "funniest and most entertaining",
        "insights": "most educational and insightful",
        "emotional": "most emotionally impactful",
    }
    style = focus_prompts.get(focus, "most engaging")

    prompt = f"""
You are a viral social media expert. Based on this transcript, create content optimized for the {style} style.

Transcript: "{transcript or 'No speech detected'}"

Return ONLY a JSON object (no markdown backticks):
{{
  "title": "Hook-driven title under 10 words",
  "description": "Two sentence platform description that drives engagement",
  "hashtags": ["#tag1", "#tag2", "#tag3", "#tag4", "#tag5"],
  "score": 87
}}
"""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{GEMINI_BASE}/models/gemini-2.0-flash:generateContent?key={GEMINI_KEY}",
            json={"contents": [{"parts": [{"text": prompt}]}]}
        )
    try:
        raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        return json.loads(raw)
    except Exception:
        return {
            "title": "Highlight clip",
            "description": "A standout moment from your video.",
            "hashtags": ["#viral", "#highlight", "#clips", "#ai", "#trending"],
            "score": 80
        }
