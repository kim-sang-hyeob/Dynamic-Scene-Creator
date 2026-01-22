import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from server.tasks import process_3dgs_job

app = FastAPI(title="3DGS Automated Service API")

# Mount static files to serve .ply results
STATIC_DIR = "server/static"
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Simple in-memory job store (In production, use Redis/Postgres)
jobs = {}

class JobStatus(BaseModel):
    id: str
    status: str # "queued", "processing", "completed", "failed"
    model: str
    scene_name: str
    result_url: Optional[str] = None
    error: Optional[str] = None

UPLOAD_DIR = "server/uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload", response_model=JobStatus)
async def upload_video(
    video: UploadFile = File(...),
    model: str = "fastgs",
    scene_name: Optional[str] = None
):
    # 1. Generate Job ID
    job_id = str(uuid.uuid4())
    if not scene_name:
        scene_name = f"scene_{job_id[:8]}"
    
    # 2. Save Video
    video_ext = os.path.splitext(video.filename)[1]
    video_path = os.path.join(UPLOAD_DIR, f"{job_id}{video_ext}")
    
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    
    # 3. Create Job Record
    job = {
        "id": job_id,
        "status": "queued",
        "model": model,
        "scene_name": scene_name,
        "video_path": video_path
    }
    jobs[job_id] = job
    
    # 4. Trigger Celery
    process_3dgs_job.delay(job_id, video_path, model, scene_name)
    print(f"[Server] Job {job_id} queued. Video: {video_path}")
    
    return job

@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.get("/jobs", response_model=List[JobStatus])
async def list_jobs():
    return list(jobs.values())

# Note: Integration with Celery/Runner will be in server/tasks.py
