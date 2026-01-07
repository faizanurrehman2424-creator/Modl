""""
AI Influencer Prompt Builder for SDXL/Flux
Updated to prioritize Framing and Pose (Fixes Headshot Issue)
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class InfluencerParams:
    """Structured input from frontend"""
    age: int
    gender: str       
    ethnicity: str    
    face_shape: str   
    hair_style: str
    hair_color: str
    eye_color: str
    body_type: str
    style_preset: str 
    scenario: Optional[str] = "portrait"
    clothing: Optional[str] = None
    expression: Optional[str] = "friendly smile"
    background: Optional[str] = None
    
    # --- NEW FIELDS FOR CAMERA CONTROL ---
    pose: Optional[str] = "standing"
    framing: Optional[str] = "waist_up"
    camera_angle: Optional[str] = "eye_level"
    # -------------------------------------
    
    garment_image_url: Optional[str] = None 
    original_job_id: Optional[str] = None


class PromptBuilder:
    """
    Main prompt builder class
    Converts structured parameters into SDXL-optimized prompts
    """
    
    def __init__(self):
        # Quality and technical tokens
        self.quality_tokens = [
            "high quality",
            "detailed",
            "8k",
            "professional photography",
            "sharp focus",
            "photorealistic"
        ]
        
        # --- NEW: FRAMING TOKENS (Crucial for fixing headshots) ---
        self.framing_tokens = {
            "full_body": "full body wide shot, entire outfit visible, head to toe, wide angle",
            "waist_up": "medium shot, waist up portrait",
            "headshot": "close-up face portrait, head and shoulders",
            "extreme_close_up": "extreme close-up macro shot of face"
        }

        # --- NEW: CAMERA ANGLE TOKENS ---
        self.camera_angle_tokens = {
            "eye_level": "eye level angle",
            "low_angle": "low angle shot looking up, empowering angle",
            "high_angle": "high angle shot looking down",
            "dutch_angle": "dutch angle, tilted camera, dynamic",
            "overhead": "overhead drone shot, top down view"
        }
        
        # Style preset configurations
        self.style_presets = {
            "fitness_influencer": {
                "environment": "modern gym, fitness studio",
                "lighting": "dramatic gym lighting, high contrast",
                "mood": "motivational, energetic, athletic",
                "composition": "dynamic pose, action shot",
                "extras": "athletic physique, toned muscles"
            },
            "fashion_blogger": {
                "environment": "urban street, city background, fashion district",
                "lighting": "golden hour, natural light, soft shadows",
                "mood": "stylish, confident, trendy",
                "composition": "fashion pose", # Removed "full body" hardcode to let user decide
                "extras": "trendy outfit, street style, fashionable"
            },
            "lifestyle": {
                "environment": "cozy home interior, cafe, outdoor park",
                "lighting": "natural window light, soft ambient lighting",
                "mood": "casual, relaxed, authentic, candid",
                "composition": "natural pose",
                "extras": "casual wear, everyday style"
            },
            "professional": {
                "environment": "office, studio, neutral background",
                "lighting": "professional studio lighting, three-point lighting",
                "mood": "confident, professional, approachable",
                "composition": "corporate portrait",
                "extras": "business attire, formal wear"
            },
            "travel_blogger": {
                "environment": "exotic location, travel destination, scenic background",
                "lighting": "natural outdoor lighting, sunset, golden hour",
                "mood": "adventurous, excited, wanderlust",
                "composition": "environmental portrait, location emphasis",
                "extras": "travel outfit, backpack, camera"
            },
            "beauty_guru": {
                "environment": "beauty studio, clean background, makeup station",
                "lighting": "ring light, beauty lighting, soft diffused light",
                "mood": "glamorous, polished, elegant",
                "composition": "beauty shot",
                "extras": "makeup, skincare, beauty products"
            },
            "tech_reviewer": {
                "environment": "modern office, tech workspace, minimalist setup",
                "lighting": "LED lighting, clean studio light",
                "mood": "innovative, professional, knowledgeable",
                "composition": "medium shot with tech products",
                "extras": "tech gadgets, modern devices"
            }
        }

        # Pose configurations
        self.pose_tokens = {
            "standing": "standing confidently",
            "sitting": "sitting relaxed",
            "walking": "walking towards camera, in motion",
            "leaning": "leaning against wall",
            "crossed_arms": "arms crossed",
            "hands_in_pockets": "hands in pockets",
            "portrait_closeup": "close-up face portrait, looking at camera",
            "full_body_standing": "full body shot, standing confidently",
            "sitting_casual": "sitting casually on a chair",
            "walking_toward_camera": "walking towards the camera",
            "side_profile": "side profile view",
            "leaning_against_wall": "leaning back against a wall",
            "crossed_arms_confident": "standing with arms crossed",
            "holding_product_placeholder": "holding an object in hand"
        }
        
        # Age descriptors
        self.age_descriptors = {
            (18, 24): "young adult",
            (25, 34): "adult",
            (35, 44): "mature adult",
            (45, 60): "middle-aged"
        }
        
        # Body type tokens
        self.body_type_tokens = {
            "slim": "slim build, lean physique",
            "athletic": "athletic build, fit physique, toned",
            "average": "average build, medium frame",
            "muscular": "muscular build, strong physique",
            "curvy": "curvy figure, hourglass shape",
            "plus_size": "plus size, full-figured"
        }
        
        # Hair style tokens
        self.hair_style_tokens = {
            "long": "long flowing hair",
            "short": "short cropped hair",
            "medium": "shoulder-length hair",
            "pixie": "pixie cut",
            "bob": "bob haircut",
            "curly": "curly hair",
            "wavy": "wavy hair",
            "straight": "straight hair",
            "braided": "braided hair",
            "ponytail": "ponytail hairstyle",
            "bun": "hair in bun"
        }
        
        # Expression tokens
        self.expression_tokens = {
            "smile": "friendly smile, warm expression",
            "serious": "serious expression, focused look",
            "confident": "confident expression, determined look",
            "playful": "playful expression, fun smile",
            "mysterious": "mysterious expression, subtle smile",
            "natural": "natural expression, relaxed face"
        }

    def build_prompt(self, params: InfluencerParams) -> str:
        """
        Build the main positive prompt
        """
        prompt_parts = []
        
        # --- 1. FRAMING & CAMERA (First Priority) ---
        # We put this first so the AI knows the composition immediately
        framing_key = params.framing.lower().replace(" ", "_") if params.framing else "waist_up"
        framing_desc = self.framing_tokens.get(framing_key, "medium shot")
        prompt_parts.append(framing_desc)

        if params.camera_angle:
            angle_key = params.camera_angle.lower().replace(" ", "_")
            angle_desc = self.camera_angle_tokens.get(angle_key, "")
            if angle_desc:
                prompt_parts.append(angle_desc)
        
        # --- 2. SUBJECT DESCRIPTION ---
        subject = self._build_subject_description(params)
        prompt_parts.append(subject)
        
        # --- 3. POSE (High Priority) ---
        if params.pose:
            # Clean key
            pose_key = params.pose.lower().replace(" ", "_")
            pose_desc = self.pose_tokens.get(pose_key, params.pose)
            # We add weight (1.3) to force the pose
            prompt_parts.append(f"({pose_desc}:1.3)")
        
        # --- 4. PHYSICAL FEATURES ---
        features = self._build_physical_features(params)
        prompt_parts.append(features)
        
        # --- 5. STYLE PRESET ELEMENTS ---
        style = self._build_style_elements(params)
        prompt_parts.append(style)
        
        # --- 6. QUALITY TOKENS ---
        quality = ", ".join(self.quality_tokens)
        prompt_parts.append(quality)
        
        return ", ".join(prompt_parts)
    
    def _build_subject_description(self, params: InfluencerParams) -> str:
        """Build the core subject description"""
        age_desc = self._get_age_descriptor(params.age)
        
        # --- FIX: REMOVED "PORTRAIT" FROM HERE ---
        # Old code: "professional portrait photograph of a..."
        # New code: "professional photograph of a..."
        # This allows the 'framing' variable (Step 1) to control the shot type.
        subject = f"professional photograph of a {age_desc} {params.gender} {params.ethnicity} person"
        return subject
    
    def _build_physical_features(self, params: InfluencerParams) -> str:
        """Build detailed physical feature description"""
        features = []
        
        features.append(f"{params.face_shape} face shape")
        
        hair_style = self.hair_style_tokens.get(params.hair_style.lower(), params.hair_style)
        features.append(f"{hair_style}, {params.hair_color} hair")
        
        features.append(f"{params.eye_color} eyes")
        
        body = self.body_type_tokens.get(params.body_type.lower(), params.body_type)
        features.append(body)
        
        expression = self.expression_tokens.get(params.expression.lower(), params.expression)
        features.append(expression)
        
        if params.clothing:
            features.append(f"wearing {params.clothing}")
        
        return ", ".join(features)
    
    def _build_style_elements(self, params: InfluencerParams) -> str:
        """Build style-specific elements"""
        # Normalize the key: "Fashion Blogger" -> "fashion_blogger"
        preset_key = params.style_preset.lower().replace(" ", "_")
        preset = self.style_presets.get(preset_key, {})
        
        style_parts = []
        
        # Environment
        if params.background:
            style_parts.append(params.background)
        elif preset.get("environment"):
            style_parts.append(preset["environment"])
        
        # Lighting
        if preset.get("lighting"):
            style_parts.append(preset["lighting"])
        
        # Mood
        if preset.get("mood"):
            style_parts.append(preset["mood"])
        
        # Composition
        # Only add preset composition if user didn't specify a pose
        if not params.pose and preset.get("composition"):
            style_parts.append(preset["composition"])
        
        # Extras
        if preset.get("extras"):
            style_parts.append(preset["extras"])
        
        # Scenario override
        if params.scenario and params.scenario != "portrait":
            style_parts.append(params.scenario)
        
        return ", ".join(style_parts)
    
    def _get_age_descriptor(self, age: int) -> str:
        """Convert age to natural language descriptor"""
        for (min_age, max_age), descriptor in self.age_descriptors.items():
            if min_age <= age <= max_age:
                return f"{age}-year-old {descriptor}"
        return f"{age}-year-old"
    
    def build_negative_prompt(self, params: Optional[InfluencerParams] = None) -> str:
        """Standard negative prompt"""
        base_negatives = [
            "blurry", "out of focus", "low quality", "low resolution",
            "pixelated", "grainy", "noisy", 
            "bad anatomy", "deformed", "disfigured", "mutation",
            "extra limbs", "extra fingers", "bad hands", "missing fingers",
            "cartoon", "anime", "painting", "drawing", "illustration", "3d render",
            "watermark", "text", "signature", "logo"
        ]
        return ", ".join(base_negatives)

    def get_prompt_analysis(self, params: InfluencerParams) -> Dict:
        """Analyze prompt for debugging"""
        return {
            "positive_prompt": self.build_prompt(params),
            "negative_prompt": self.build_negative_prompt(params),
            "subject": self._build_subject_description(params),
            "total_length": len(self.build_prompt(params))
        }

# Testing block
if __name__ == "__main__":
    builder = PromptBuilder()
    params = InfluencerParams(
        age=25,
        gender="female",
        ethnicity="asian",
        face_shape="oval",
        hair_style="long",
        hair_color="black",
        eye_color="brown",
        body_type="slim",
        style_preset="fashion_blogger",
        pose="full_body_standing",
        clothing="red dress",
        framing="full_body",
        camera_angle="low_angle"
    )
    print(builder.build_prompt(params))
