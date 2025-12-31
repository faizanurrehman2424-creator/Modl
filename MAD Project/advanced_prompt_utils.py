"""
Advanced prompt utilities for optimization and enhancement
"""

from typing import List, Dict, Tuple, Optional
import re
from collections import Counter


class PromptOptimizer:
    """
    Optimize prompts for better SDXL results
    Handle token limits, redundancy, and ordering
    """
    
    # SDXL performs better with certain token orderings
    PRIORITY_TOKENS = [
        "professional", "portrait", "photograph", "photo",
        "high quality", "detailed", "8k", "photorealistic"
    ]
    
    # Tokens that conflict or are redundant
    CONFLICTING_PAIRS = [
        ("cartoon", "photorealistic"),
        ("illustration", "photograph"),
        ("painting", "photo"),
        ("anime", "realistic")
    ]
    
    def __init__(self, max_tokens: int = 150):
        self.max_tokens = max_tokens
    
    def optimize(self, prompt: str) -> str:
        """
        Optimize prompt for better results
        
        Args:
            prompt: Raw prompt string
            
        Returns:
            Optimized prompt
        """
        # Split into tokens
        tokens = self._tokenize(prompt)
        
        # Remove duplicates while preserving order
        tokens = self._remove_duplicates(tokens)
        
        # Remove conflicting tokens
        tokens = self._remove_conflicts(tokens)
        
        # Reorder for priority
        tokens = self._reorder_by_priority(tokens)
        
        # Truncate if too long
        if len(tokens) > self.max_tokens:
            tokens = tokens[:self.max_tokens]
        
        # Rejoin
        return ", ".join(tokens)
    
    def _tokenize(self, prompt: str) -> List[str]:
        """Split prompt into tokens"""
        # Split by comma and strip whitespace
        tokens = [t.strip() for t in prompt.split(",")]
        return [t for t in tokens if t]  # Remove empty
    
    def _remove_duplicates(self, tokens: List[str]) -> List[str]:
        """Remove duplicate tokens while preserving order"""
        seen = set()
        result = []
        for token in tokens:
            token_lower = token.lower()
            if token_lower not in seen:
                seen.add(token_lower)
                result.append(token)
        return result
    
    def _remove_conflicts(self, tokens: List[str]) -> List[str]:
        """Remove conflicting tokens, keeping the first occurrence"""
        tokens_lower = [t.lower() for t in tokens]
        
        for conflict_a, conflict_b in self.CONFLICTING_PAIRS:
            if conflict_a in tokens_lower and conflict_b in tokens_lower:
                # Remove the second occurrence
                idx_a = tokens_lower.index(conflict_a)
                idx_b = tokens_lower.index(conflict_b)
                
                if idx_b > idx_a:
                    tokens.pop(idx_b)
                    tokens_lower.pop(idx_b)
                else:
                    tokens.pop(idx_a)
                    tokens_lower.pop(idx_a)
        
        return tokens
    
    def _reorder_by_priority(self, tokens: List[str]) -> List[str]:
        """Move priority tokens to front"""
        priority = []
        regular = []
        
        tokens_lower = [t.lower() for t in tokens]
        
        for token in tokens:
            if any(p in token.lower() for p in self.PRIORITY_TOKENS):
                priority.append(token)
            else:
                regular.append(token)
        
        return priority + regular
    
    def analyze_prompt_strength(self, prompt: str) -> Dict:
        """
        Analyze prompt for potential issues
        
        Returns:
            Dictionary with analysis metrics
        """
        tokens = self._tokenize(prompt)
        
        # Count token types
        quality_tokens = sum(1 for t in tokens if any(
            q in t.lower() for q in ["quality", "detailed", "sharp", "8k", "4k", "photorealistic"]
        ))
        
        subject_tokens = sum(1 for t in tokens if any(
            s in t.lower() for s in ["person", "woman", "man", "portrait", "face"]
        ))
        
        style_tokens = sum(1 for t in tokens if any(
            s in t.lower() for s in ["lighting", "background", "studio", "environment"]
        ))
        
        # Check for issues
        issues = []
        if len(tokens) < 10:
            issues.append("Prompt too short - add more descriptive tokens")
        if len(tokens) > 100:
            issues.append("Prompt too long - may get truncated")
        if quality_tokens == 0:
            issues.append("No quality tokens - add 'high quality', '8k', etc.")
        if subject_tokens == 0:
            issues.append("No clear subject - add person description")
        
        return {
            "total_tokens": len(tokens),
            "quality_tokens": quality_tokens,
            "subject_tokens": subject_tokens,
            "style_tokens": style_tokens,
            "estimated_strength": self._estimate_strength(tokens),
            "issues": issues
        }
    
    def _estimate_strength(self, tokens: List[str]) -> float:
        """Estimate prompt strength (0-1)"""
        score = 0.0
        
        # Length score (optimal around 30-50 tokens)
        length_score = min(len(tokens) / 40, 1.0)
        score += length_score * 0.3
        
        # Quality token score
        quality_count = sum(1 for t in tokens if "quality" in t.lower() or "detailed" in t.lower())
        quality_score = min(quality_count / 3, 1.0)
        score += quality_score * 0.3
        
        # Subject clarity score
        subject_words = ["person", "woman", "man", "portrait", "face", "body"]
        subject_score = min(sum(1 for t in tokens if any(s in t.lower() for s in subject_words)) / 5, 1.0)
        score += subject_score * 0.4
        
        return score


class PromptWeightManager:
    """
    Manage token weights for SDXL
    SDXL supports (token:weight) syntax
    """
    
    def __init__(self):
        # Default weights for different token categories
        self.category_weights = {
            "subject": 1.2,      # Emphasize the main subject
            "quality": 1.1,      # Emphasize quality
            "style": 1.0,        # Normal weight
            "background": 0.9    # De-emphasize background
        }
    
    def apply_weights(
        self, 
        tokens: List[str], 
        categories: Dict[str, List[str]]
    ) -> str:
        """
        Apply weights to tokens based on categories
        
        Args:
            tokens: List of prompt tokens
            categories: Dict mapping category names to token indices
            
        Returns:
            Weighted prompt string
        """
        weighted_tokens = []
        
        for i, token in enumerate(tokens):
            weight = self._get_token_weight(i, categories)
            
            if weight != 1.0:
                weighted_tokens.append(f"({token}:{weight})")
            else:
                weighted_tokens.append(token)
        
        return ", ".join(weighted_tokens)
    
    def _get_token_weight(self, index: int, categories: Dict[str, List[str]]) -> float:
        """Get weight for token at index"""
        for category, indices in categories.items():
            if index in indices:
                return self.category_weights.get(category, 1.0)
        return 1.0
    
    def emphasize_tokens(self, prompt: str, tokens_to_emphasize: List[str], weight: float = 1.3) -> str:
        """
        Emphasize specific tokens in prompt
        
        Args:
            prompt: Original prompt
            tokens_to_emphasize: List of tokens to emphasize
            weight: Weight to apply (default 1.3)
            
        Returns:
            Modified prompt with emphasized tokens
        """
        for token in tokens_to_emphasize:
            # Case-insensitive replacement
            pattern = re.compile(re.escape(token), re.IGNORECASE)
            prompt = pattern.sub(f"({token}:{weight})", prompt)
        
        return prompt


class PromptVariationGenerator:
    """
    Generate variations of a prompt for diverse results
    """
    
    LIGHTING_VARIATIONS = [
        "golden hour lighting",
        "soft window light",
        "dramatic studio lighting",
        "natural outdoor lighting",
        "ring light beauty lighting",
        "moody low-key lighting"
    ]
    
    ANGLE_VARIATIONS = [
        "shot from slightly above",
        "eye level shot",
        "shot from below",
        "three-quarter view",
        "side profile view"
    ]
    
    EXPRESSION_VARIATIONS = [
        "friendly smile",
        "confident expression",
        "playful look",
        "serious professional expression",
        "natural relaxed expression"
    ]
    
    def generate_variations(
        self, 
        base_prompt: str, 
        num_variations: int = 4,
        variation_types: List[str] = ["lighting", "angle", "expression"]
    ) -> List[str]:
        """
        Generate multiple prompt variations
        
        Args:
            base_prompt: Base prompt to vary
            num_variations: Number of variations to generate
            variation_types: Types of variations to apply
            
        Returns:
            List of varied prompts
        """
        variations = [base_prompt]  # Include original
        
        for i in range(num_variations - 1):
            varied = base_prompt
            
            if "lighting" in variation_types and i < len(self.LIGHTING_VARIATIONS):
                varied = self._add_or_replace_lighting(varied, self.LIGHTING_VARIATIONS[i])
            
            if "angle" in variation_types and i < len(self.ANGLE_VARIATIONS):
                varied = f"{varied}, {self.ANGLE_VARIATIONS[i]}"
            
            if "expression" in variation_types and i < len(self.EXPRESSION_VARIATIONS):
                varied = self._replace_expression(varied, self.EXPRESSION_VARIATIONS[i])
            
            variations.append(varied)
        
        return variations
    
    def _add_or_replace_lighting(self, prompt: str, new_lighting: str) -> str:
        """Replace lighting in prompt or add if not present"""
        # Common lighting keywords
        lighting_keywords = ["lighting", "light", "lit", "illumination"]
        
        tokens = prompt.split(", ")
        found_lighting = False
        
        for i, token in enumerate(tokens):
            if any(kw in token.lower() for kw in lighting_keywords):
                tokens[i] = new_lighting
                found_lighting = True
                break
        
        if not found_lighting:
            tokens.append(new_lighting)
        
        return ", ".join(tokens)
    
    def _replace_expression(self, prompt: str, new_expression: str) -> str:
        """Replace expression in prompt"""
        expression_keywords = ["smile", "expression", "look", "facial"]
        
        tokens = prompt.split(", ")
        
        for i, token in enumerate(tokens):
            if any(kw in token.lower() for kw in expression_keywords):
                tokens[i] = new_expression
                return ", ".join(tokens)
        
        # If no expression found, add it
        tokens.append(new_expression)
        return ", ".join(tokens)


class PromptTemplateLibrary:
    """
    Pre-built templates for common use cases
    """
    
    TEMPLATES = {
        "instagram_model": """
            professional instagram photo of {subject}, {features}, 
            trendy outfit, lifestyle photography, golden hour lighting,
            bokeh background, shallow depth of field, candid moment,
            influencer aesthetic, high quality, 8k
        """,
        
        "linkedin_professional": """
            professional linkedin headshot of {subject}, {features},
            business attire, corporate setting, studio lighting,
            neutral gray background, confident expression,
            professional photography, sharp focus, high quality
        """,
        
        "fitness_post": """
            fitness influencer photo of {subject}, {features},
            athletic wear, gym environment, workout scene,
            motivational, energetic, dynamic lighting,
            athletic photography, high contrast, 8k detailed
        """,
        
        "beauty_closeup": """
            beauty photography portrait of {subject}, {features},
            makeup, flawless skin, ring light, clean background,
            close-up shot, beauty lighting, glamorous,
            professional beauty photography, high detail, 8k
        """,
        
        "casual_lifestyle": """
            casual lifestyle photo of {subject}, {features},
            everyday wear, home interior, natural window light,
            relaxed atmosphere, authentic moment, candid photography,
            lifestyle blogger aesthetic, natural colors, high quality
        """
    }
    
    def get_template(
        self, 
        template_name: str, 
        subject: str, 
        features: str
    ) -> str:
        """
        Fill in a template with subject and features
        
        Args:
            template_name: Name of template to use
            subject: Subject description
            features: Physical features description
            
        Returns:
            Filled template as prompt
        """
        template = self.TEMPLATES.get(template_name, "")
        
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        # Fill in template
        prompt = template.format(subject=subject, features=features)
        
        # Clean up whitespace
        prompt = " ".join(prompt.split())
        
        return prompt
    
    def list_templates(self) -> List[str]:
        """Get list of available templates"""
        return list(self.TEMPLATES.keys())


# Example usage
if __name__ == "__main__":
    print("=== Prompt Optimization ===")
    optimizer = PromptOptimizer()
    
    messy_prompt = """
        portrait, photo, photograph, professional portrait, 
        woman, young woman, detailed, high quality, detailed face,
        cartoon style, photorealistic, 8k, 8k resolution
    """
    
    optimized = optimizer.optimize(messy_prompt)
    print("Original:", messy_prompt)
    print("Optimized:", optimized)
    print()
    
    analysis = optimizer.analyze_prompt_strength(optimized)
    print("Analysis:", analysis)
    print()
    
    print("=== Prompt Variations ===")
    variation_gen = PromptVariationGenerator()
    
    base = "professional portrait of a 28-year-old woman, blonde hair, blue eyes, athletic build"
    variations = variation_gen.generate_variations(base, num_variations=3)
    
    for i, var in enumerate(variations, 1):
        print(f"Variation {i}: {var}")
    print()
    
    print("=== Template Library ===")
    templates = PromptTemplateLibrary()
    
    subject = "25-year-old woman"
    features = "blonde hair, blue eyes, oval face, athletic build"
    
    for template_name in templates.list_templates():
        print(f"\n{template_name.upper()}:")
        prompt = templates.get_template(template_name, subject, features)
        print(prompt[:150] + "...")  # Show first 150 chars