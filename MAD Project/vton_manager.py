"""
Virtual Try-On (VTON) Manager
Handles the integration with Replicate's Try-On models
"""

import replicate
import os
from typing import Optional

class VTONManager:
    """
    Manages Virtual Try-On operations
    Uses 'cuuupid/idm-vton' or similar reliable model on Replicate
    """
    
    def __init__(self):
        self.api_token = os.getenv("REPLICATE_API_TOKEN")
        # Using a reliable IDM-VTON model version
        self.model_version = "cuuupid/idm-vton:c871bb9b0466074280c2a9a7386749c8b0fbfd26b5e61e366e81a2e134626729"

    async def apply_clothing(
        self, 
        person_image_url: str, 
        garment_image_url: str,
        category: str = "upper_body" # upper_body, lower_body, dresses
    ) -> str:
        """
        Apply the garment to the person image
        """
        if not self.api_token:
            raise Exception("REPLICATE_API_TOKEN is not set")

        print(f"👕 Starting Virtual Try-On...")
        print(f"   Person: {person_image_url[:30]}...")
        print(f"   Garment: {garment_image_url[:30]}...")

        try:
            # Run the IDM-VTON model
            # Note: Input parameters vary slightly by model version.
            # This is standard for IDM-VTON on Replicate.
            output = replicate.run(
                self.model_version,
                input={
                    "human_img": person_image_url,
                    "garm_img": garment_image_url,
                    "garment_des": "clothing", # Optional description
                    "category": category,
                    "crop": False,
                    "seed": 42,
                    "steps": 30
                }
            )
            
            # Replicate usually returns a string URL or a list containing the URL
            result_url = output if isinstance(output, str) else output[0]
            print(f"✅ VTON Success: {result_url[:30]}...")
            return result_url

        except Exception as e:
            print(f"❌ VTON Failed: {str(e)}")
            raise Exception(f"Virtual Try-On failed: {str(e)}")