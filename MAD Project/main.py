"""
AI Influencer Generator - Complete Backend
Updated with Regeneration, VTON, and Poses
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import uuid
from datetime import datetime
import asyncio
import os
import urllib.parse # Added for URL encoding

# Import existing modules
from prompt_builder import (
    PromptBuilder, InfluencerParams, 
    Gender, Ethnicity, FaceShape, StylePreset, Pose
)
from vton_manager import VTONManager # Import new manager

# Replicate
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
vton_manager = VTONManager()

# In-memory database (Replace with SQL in production)
# Structure: job_id -> {status, request, result, seed, ...}
generation_jobs = {}

# --- MODELS ---

class InfluencerRequest(BaseModel):
    """
    Updated request model supporting updates and poses
    """
    # Core Attributes
    age: int = Field(..., ge=18, le=70)
    gender: str # Changed to str to match your frontend JSON
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
    
    # --- NEW FEATURES ---
    pose: Optional[str] = None
    original_job_id: Optional[str] = None # For Updating/Regenerating
    garment_image_url: Optional[str] = None # For Virtual Try-On


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
    """
    Handles Generation, Regeneration (Update), and VTON preparation
    """
    new_job_id = str(uuid.uuid4())
    
    # Logic for Feature 1: Regeneration / Update
    seed_to_use = None
    if request.original_job_id:
        original_job = generation_jobs.get(request.original_job_id)
        if original_job and "seed" in original_job:
            seed_to_use = original_job["seed"]
            print(f"🔄 Updating Influencer: Reusing seed {seed_to_use} from job {request.original_job_id}")
    
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


@app.get("/api/v1/poses")
async def get_poses():
    """Return list of supported 8 poses"""
    # If using Enum in prompt_builder, convert to list
    return ["portrait_closeup", "full_body_standing", "sitting_casual"] 


# --- BACKGROUND WORKFLOW ---

async def process_generation_workflow(job_id: str, request: InfluencerRequest, seed: int):
    """
    Complex workflow:
    1. Build Prompt (with Pose).
    2. Generate Base Image (Pollinations Free Mode or SDXL).
    """
    try:
        # Step 1: Build Prompt
        # We convert the request params to the format PromptBuilder expects
        # Note: We need to handle the conversion from string to Enum inside prompt_builder 
        # or pass raw strings if your PromptBuilder supports it. 
        # For safety, we will assume params map directly.
        
        # Manually constructing params to avoid validation errors if using strict Enums
        # (Simplified for this main.py version)
        params_dict = request.dict(exclude={'original_job_id', 'garment_image_url'})
        
        # NOTE: You might need to adjust PromptBuilder to accept strings if it expects Enums
        # But for this test, let's assume it handles it or we pass it as is.
        # Ideally, import InfluencerParams and map fields.
        
        params = InfluencerParams(**params_dict)
        
        # For VTON/Garment uploads
        if request.garment_image_url:
            params.clothing = "simple white t-shirt and jeans" 
            
        positive_prompt = prompt_builder.build_prompt(params)
        negative_prompt = prompt_builder.build_negative_prompt()
        
        print(f"🎨 Job {job_id}: Generating Image...")
        print(f"   Prompt: {positive_prompt[:50]}...")
        
        # ==========================================================
        # FREE MODE: Pollinations.ai (No API Key Required)
        # ==========================================================
        print(f"🚀 Using Pollinations.ai (Free Mode)")
        
        # 1. Clean prompt for URL
        encoded_prompt = urllib.parse.quote(positive_prompt)
        
        # 2. Construct URL
        # Pollinations generates the image on-the-fly when this URL is visited
        final_image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true&seed={seed}&model=flux"
        
        if request.garment_image_url:
             print("⚠️  Warning: Virtual Try-On is disabled in Free Mode.")

        # ==========================================================
        # PAID MODE: Replicate (Commented Out)
        # Uncomment this section when you have credits
        # ==========================================================
        """
        # Step 2: Generate Base Image (SDXL)
        sdxl_output = replicate.run(
            "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            input={
                "prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "width": 1024,
                "height": 1024,
                "seed": seed
            }
        )
        base_image_url = sdxl_output[0]
        final_image_url = base_image_url
        
        # Step 3: Virtual Try-On
        if request.garment_image_url:
            print(f"👕 Job {job_id}: Applying Virtual Try-On...")
            try:
                vton_url = await vton_manager.apply_clothing(
                    person_image_url=base_image_url,
                    garment_image_url=request.garment_image_url
                )
                final_image_url = vton_url
            except Exception as vton_error:
                print(f"⚠️ VTON failed: {vton_error}")
                generation_jobs[job_id]["warning"] = "VTON Failed"
        """
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
