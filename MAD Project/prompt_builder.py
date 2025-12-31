"""
AI Influencer Prompt Builder for SDXL
Converts structured user input into optimized SDXL prompts
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    NON_BINARY = "non-binary"


class FaceShape(str, Enum):
    OVAL = "oval"
    ROUND = "round"
    SQUARE = "square"
    HEART = "heart-shaped"
    LONG = "long"
    DIAMOND = "diamond"


class Ethnicity(str, Enum):
    CAUCASIAN = "caucasian"
    ASIAN = "asian"
    AFRICAN = "african"
    HISPANIC = "hispanic"
    MIDDLE_EASTERN = "middle eastern"
    MIXED = "mixed"


class StylePreset(str, Enum):
    FITNESS = "fitness_influencer"
    FASHION = "fashion_blogger"
    LIFESTYLE = "lifestyle"
    PROFESSIONAL = "professional"
    TRAVEL = "travel_blogger"
    BEAUTY = "beauty_guru"
    TECH = "tech_reviewer"


class Pose(str, Enum):
    """Supported poses for the influencer"""
    PORTRAIT_CLOSEUP = "portrait_closeup"
    FULL_BODY_STANDING = "full_body_standing"
    SITTING_CASUAL = "sitting_casual"
    WALKING_TOWARD_CAMERA = "walking_toward_camera"
    SIDE_PROFILE = "side_profile"
    LEANING_AGAINST_WALL = "leaning_against_wall"
    CROSSED_ARMS = "crossed_arms_confident"
    HOLDING_PRODUCT = "holding_product_placeholder"


@dataclass
class InfluencerParams:
    """Structured input from frontend"""
    age: int
    gender: Gender
    ethnicity: Ethnicity
    face_shape: FaceShape
    hair_style: str
    hair_color: str
    eye_color: str
    body_type: str
    style_preset: StylePreset
    scenario: Optional[str] = "portrait"
    clothing: Optional[str] = None
    expression: Optional[str] = "friendly smile"
    background: Optional[str] = None
    pose: Optional[Pose] = None


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
        
        # Style preset configurations
        self.style_presets = {
            StylePreset.FITNESS: {
                "environment": "modern gym, fitness studio",
                "lighting": "dramatic gym lighting, high contrast",
                "mood": "motivational, energetic, athletic",
                "composition": "dynamic pose, action shot",
                "extras": "athletic physique, toned muscles"
            },
            StylePreset.FASHION: {
                "environment": "urban street, city background, fashion district",
                "lighting": "golden hour, natural light, soft shadows",
                "mood": "stylish, confident, trendy",
                "composition": "full body shot, fashion pose",
                "extras": "trendy outfit, street style, fashionable"
            },
            StylePreset.LIFESTYLE: {
                "environment": "cozy home interior, cafe, outdoor park",
                "lighting": "natural window light, soft ambient lighting",
                "mood": "casual, relaxed, authentic, candid",
                "composition": "medium shot, natural pose",
                "extras": "casual wear, everyday style"
            },
            StylePreset.PROFESSIONAL: {
                "environment": "office, studio, neutral background",
                "lighting": "professional studio lighting, three-point lighting",
                "mood": "confident, professional, approachable",
                "composition": "headshot, corporate portrait",
                "extras": "business attire, formal wear"
            },
            StylePreset.TRAVEL: {
                "environment": "exotic location, travel destination, scenic background",
                "lighting": "natural outdoor lighting, sunset, golden hour",
                "mood": "adventurous, excited, wanderlust",
                "composition": "environmental portrait, location emphasis",
                "extras": "travel outfit, backpack, camera"
            },
            StylePreset.BEAUTY: {
                "environment": "beauty studio, clean background, makeup station",
                "lighting": "ring light, beauty lighting, soft diffused light",
                "mood": "glamorous, polished, elegant",
                "composition": "close-up portrait, beauty shot",
                "extras": "makeup, skincare, beauty products"
            },
            StylePreset.TECH: {
                "environment": "modern office, tech workspace, minimalist setup",
                "lighting": "LED lighting, clean studio light",
                "mood": "innovative, professional, knowledgeable",
                "composition": "medium shot with tech products",
                "extras": "tech gadgets, modern devices"
            }
        }

        # Pose configurations (New Feature)
        self.pose_tokens = {
            Pose.PORTRAIT_CLOSEUP: "close-up face portrait, looking at camera, head and shoulders shot",
            Pose.FULL_BODY_STANDING: "full body shot, standing confidently, facing camera, fashion pose, entire outfit visible, wide angle",
            Pose.SITTING_CASUAL: "sitting casually on a chair, relaxed posture, knee up, lifestyle photography",
            Pose.WALKING_TOWARD_CAMERA: "walking towards the camera, dynamic movement, street style photography, in motion",
            Pose.SIDE_PROFILE: "side profile view, looking into distance, turning head, artistic angle",
            Pose.LEANING_AGAINST_WALL: "leaning back against a wall, cool attitude, relaxed stance, fashion editorial pose",
            Pose.CROSSED_ARMS: "standing with arms crossed, powerful stance, boss energy, professional posture",
            Pose.HOLDING_PRODUCT: "holding an object in hand, presenting to camera, focus on hands, promotional pose"
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
        
        # 1. Subject description (Most important)
        subject = self._build_subject_description(params)
        prompt_parts.append(subject)
        
        # 2. Physical features
        features = self._build_physical_features(params)
        prompt_parts.append(features)
        
        # 3. Pose (New Feature)
        if params.pose:
            pose_desc = self.pose_tokens.get(params.pose)
            if pose_desc:
                # We weight the pose slightly higher to ensure SDXL respects it
                prompt_parts.append(f"({pose_desc}:1.3)")
        
        # 4. Style preset elements
        style = self._build_style_elements(params)
        prompt_parts.append(style)
        
        # 5. Quality tokens
        quality = ", ".join(self.quality_tokens)
        prompt_parts.append(quality)
        
        return ", ".join(prompt_parts)
    
    def _build_subject_description(self, params: InfluencerParams) -> str:
        """Build the core subject description"""
        age_desc = self._get_age_descriptor(params.age)
        
        subject = f"professional portrait photograph of a {age_desc} {params.gender.value} {params.ethnicity.value} person"
        return subject
    
    def _build_physical_features(self, params: InfluencerParams) -> str:
        """Build detailed physical feature description"""
        features = []
        
        features.append(f"{params.face_shape.value} face shape")
        
        hair_style = self.hair_style_tokens.get(params.hair_style, params.hair_style)
        features.append(f"{hair_style}, {params.hair_color} hair")
        
        features.append(f"{params.eye_color} eyes")
        
        body = self.body_type_tokens.get(params.body_type, params.body_type)
        features.append(body)
        
        expression = self.expression_tokens.get(params.expression, params.expression)
        features.append(expression)
        
        if params.clothing:
            features.append(f"wearing {params.clothing}")
        
        return ", ".join(features)
    
    def _build_style_elements(self, params: InfluencerParams) -> str:
        """Build style-specific elements"""
        preset = self.style_presets.get(params.style_preset, {})
        
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
        
        # Composition (Only add if no specific pose was selected to avoid conflict)
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
        gender=Gender.FEMALE,
        ethnicity=Ethnicity.ASIAN,
        face_shape=FaceShape.OVAL,
        hair_style="long",
        hair_color="black",
        eye_color="brown",
        body_type="slim",
        style_preset=StylePreset.FASHION,
        pose=Pose.FULL_BODY_STANDING,
        clothing="red dress"
    )
    print(builder.build_prompt(params))