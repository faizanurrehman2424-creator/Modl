"""
AI Influencer Generator - Complete Backend
Updated for Pollinations (Free Mode) & Simplified Text Inputs
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uuid
from datetime import datetime
import urllib.parse 

# FIXED IMPORT: Removed Gender, Ethnicity, etc. because they are now just text strings
from prompt_builder import PromptBuilder, InfluencerParams

# Replicate (We keep this import for when you upgrade later)
import replicate

app = FastAPI(title="AI Influencer Generator API (v2)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Systems
prompt_builder = PromptBuilder()

# In-memory database
generation_jobs = {}

# --- MODELS ---

class InfluencerRequest(BaseModel):
    """
    Request model matching the Android App JSON
    All Enums are now simple strings to prevent validation errors
    """
    age: int = Field(..., ge=18, le=70)
    gender: str
    ethnicity: str
    face_shape: str
    hair_style: str
    hair_color: str
    eye_color: str
    body_type: str
    style_preset: str
    
    # Optional / Context
    scenario: Optional[str] = "portrait"
    clothing: Optional[str] = None
    expression: Optional[str] = "friendly smile"
    background: Optional[str] = None
    
    # New Features
    pose: Optional[str] = None
    original_job_id: Optional[str] = None
    garment_image_url: Optional[str] = None


class GenerationResponse(BaseModel):
    job_id: str
    status: str
    image_url: Optional[str] = None
    created_at: str
    error: Optional[str] = None


# --- ENDPOINTS ---

@app.post("/api/v1/generate", response_model=GenerationResponse)
async def generate_influencer(
    request: InfluencerRequest,
    background_tasks: BackgroundTasks
):
    new_job_id = str(uuid.uuid4())
    
    # Logic for Consistency (Seeds)
    seed_to_use = None
    if request.original_job_id:
        original_job = generation_jobs.get(request.original_job_id)
        if original_job and "seed" in original_job:
            seed_to_use = original_job["seed"]
            print(f"🔄 Updating Influencer: Reusing seed {seed_to_use}")
    
    if not seed_to_use:
        import random
        seed_to_use = random.randint(1, 999999999)

    # Create Job Entry
    generation_jobs[new_job_id] = {
        "status": "processing",
        "created_at": datetime.utcnow().isoformat(),
        "request": request.dict(),
        "seed": seed_to_use
    }
    
    # Start Background Task
    background_tasks.add_task(
        process_generation_workflow, 
        new_job_id, 
        request, 
        seed_to_use
    )
    
    return GenerationResponse(
        job_id=new_job_id,
        status="processing",
        created_at=generation_jobs[new_job_id]["created_at"]
    )


@app.get("/api/v1/status/{job_id}", response_model=GenerationResponse)
async def get_status(job_id: str):
    if job_id not in generation_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = generation_jobs[job_id]
    return GenerationResponse(
        job_id=job_id,
        status=job["status"],
        image_url=job.get("image_url"),
        created_at=job["created_at"],
        error=job.get("error")
    )


# --- BACKGROUND WORKFLOW ---

async def process_generation_workflow(job_id: str, request: InfluencerRequest, seed: int):
    try:
        # Convert request to params
        params_dict = request.dict(exclude={'original_job_id', 'garment_image_url'})
        params = InfluencerParams(**params_dict)
        
        # Build Prompts
        positive_prompt = prompt_builder.build_prompt(params)
        
        print(f"🎨 Job {job_id}: Generating Image...")
        print(f"   Prompt: {positive_prompt[:50]}...")
        
        # ==========================================================
        # FREE MODE: Pollinations.ai (No API Key Required)
        # ==========================================================
        print(f"🚀 Using Pollinations.ai (Free Mode)")
        
        # Clean prompt for URL
        encoded_prompt = urllib.parse.quote(positive_prompt)
        
        # Construct URL
        # We add 'flux' model and seed for consistency
        final_image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true&seed={seed}&model=flux"
        
        # ==========================================================

        # Update Job
        generation_jobs[job_id].update({
            "status": "completed",
            "image_url": final_image_url
        })
        print(f"✅ Job {job_id} Completed: {final_image_url}")

    except Exception as e:
        print(f"❌ Job {job_id} Failed: {e}")
        generation_jobs[job_id].update({
            "status": "failed",
            "error": str(e)
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
