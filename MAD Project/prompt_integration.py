"""
Complete Prompt System Integration
Combines all prompt components into a unified interface
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import hashlib


@dataclass
class GenerationRequest:
    """Complete generation request with all parameters"""
    # User parameters
    user_id: str
    params: Dict
    
    # Generation options
    num_variations: int = 1
    use_template: Optional[str] = None
    variation_types: List[str] = None
    emphasize_features: List[str] = None
    
    # Advanced options
    apply_optimization: bool = True
    apply_weights: bool = False
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.variation_types is None:
            self.variation_types = ["lighting", "angle"]
        if self.emphasize_features is None:
            self.emphasize_features = []


@dataclass
class PromptResult:
    """Result containing all generated prompts and metadata"""
    positive_prompt: str
    negative_prompt: str
    variations: List[str]
    metadata: Dict
    hash: str  # For caching and consistency
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "positive_prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "variations": self.variations,
            "metadata": self.metadata,
            "hash": self.hash
        }


class PromptSystemManager:
    """
    Main manager class that coordinates all prompt operations
    This is what you'll use in your API endpoints
    """
    
    def __init__(self):
        from prompt_builder import PromptBuilder, InfluencerParams
        from advanced_prompt_utils import (
            PromptOptimizer, 
            PromptVariationGenerator,
            PromptWeightManager,
            PromptTemplateLibrary
        )
        
        self.builder = PromptBuilder()
        self.optimizer = PromptOptimizer()
        self.variation_generator = PromptVariationGenerator()
        self.weight_manager = PromptWeightManager()
        self.template_library = PromptTemplateLibrary()
        
        # Cache for repeated requests
        self.cache = {}
    
    def generate_prompts(self, request: GenerationRequest) -> PromptResult:
        """
        Main method to generate all prompts for a request
        
        Args:
            request: GenerationRequest object
            
        Returns:
            PromptResult with all prompts and metadata
        """
        # Check cache first
        cache_key = self._generate_cache_key(request)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Convert dict params to InfluencerParams
        from prompt_builder import InfluencerParams
        params = self._dict_to_params(request.params)
        
        # Build base prompt
        if request.use_template:
            positive_prompt = self._build_from_template(params, request.use_template)
        else:
            positive_prompt = self.builder.build_prompt(params)
        
        # Apply optimization if requested
        if request.apply_optimization:
            positive_prompt = self.optimizer.optimize(positive_prompt)
        
        # Apply weights if requested
        if request.apply_weights and request.emphasize_features:
            positive_prompt = self.weight_manager.emphasize_tokens(
                positive_prompt,
                request.emphasize_features,
                weight=1.3
            )
        
        # Build negative prompt
        negative_prompt = self.builder.build_negative_prompt(params)
        
        # Generate variations if requested
        variations = []
        if request.num_variations > 1:
            variations = self.variation_generator.generate_variations(
                positive_prompt,
                num_variations=request.num_variations,
                variation_types=request.variation_types
            )
        else:
            variations = [positive_prompt]
        
        # Analyze prompt quality
        analysis = self.optimizer.analyze_prompt_strength(positive_prompt)
        
        # Create metadata
        metadata = {
            "params": asdict(params),
            "seed": request.seed,
            "analysis": analysis,
            "optimized": request.apply_optimization,
            "weighted": request.apply_weights,
            "num_variations": len(variations)
        }
        
        # Create result
        result = PromptResult(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            variations=variations,
            metadata=metadata,
            hash=cache_key
        )
        
        # Cache result
        self.cache[cache_key] = result
        
        return result
    
    def _dict_to_params(self, params_dict: Dict):
        """Convert dictionary to InfluencerParams object"""
        from prompt_builder import (
            InfluencerParams, Gender, Ethnicity, 
            FaceShape, StylePreset
        )
        
        # Convert enum strings to enum values
        if 'gender' in params_dict:
            params_dict['gender'] = Gender(params_dict['gender'])
        if 'ethnicity' in params_dict:
            params_dict['ethnicity'] = Ethnicity(params_dict['ethnicity'])
        if 'face_shape' in params_dict:
            params_dict['face_shape'] = FaceShape(params_dict['face_shape'])
        if 'style_preset' in params_dict:
            params_dict['style_preset'] = StylePreset(params_dict['style_preset'])
        
        return InfluencerParams(**params_dict)
    
    def _build_from_template(self, params, template_name: str) -> str:
        """Build prompt from template"""
        subject = self.builder._build_subject_description(params)
        features = self.builder._build_physical_features(params)
        
        return self.template_library.get_template(
            template_name,
            subject,
            features
        )
    
    def _generate_cache_key(self, request: GenerationRequest) -> str:
        """Generate cache key from request"""
        # Create deterministic string from request
        key_data = {
            "params": request.params,
            "template": request.use_template,
            "optimization": request.apply_optimization,
            "weights": request.apply_weights,
            "seed": request.seed
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def validate_params(self, params: Dict) -> Tuple[bool, List[str]]:
        """
        Validate parameters before generation
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Required fields
        required = ['age', 'gender', 'ethnicity', 'face_shape', 
                   'hair_style', 'hair_color', 'eye_color', 
                   'body_type', 'style_preset']
        
        for field in required:
            if field not in params:
                errors.append(f"Missing required field: {field}")
        
        # Age validation
        if 'age' in params:
            age = params['age']
            if not isinstance(age, int) or age < 18 or age > 70:
                errors.append("Age must be between 18 and 70")
        
        # Enum validation
        from prompt_builder import Gender, Ethnicity, FaceShape, StylePreset
        
        if 'gender' in params:
            try:
                Gender(params['gender'])
            except ValueError:
                errors.append(f"Invalid gender: {params['gender']}")
        
        if 'ethnicity' in params:
            try:
                Ethnicity(params['ethnicity'])
            except ValueError:
                errors.append(f"Invalid ethnicity: {params['ethnicity']}")
        
        if 'face_shape' in params:
            try:
                FaceShape(params['face_shape'])
            except ValueError:
                errors.append(f"Invalid face_shape: {params['face_shape']}")
        
        if 'style_preset' in params:
            try:
                StylePreset(params['style_preset'])
            except ValueError:
                errors.append(f"Invalid style_preset: {params['style_preset']}")
        
        return len(errors) == 0, errors
    
    def get_prompt_suggestions(self, params: Dict) -> List[str]:
        """
        Get suggestions to improve prompt based on parameters
        """
        suggestions = []
        
        # Check if style preset matches other params
        style = params.get('style_preset', '')
        
        if style == 'fitness_influencer':
            if params.get('body_type') not in ['athletic', 'muscular']:
                suggestions.append("Consider 'athletic' or 'muscular' body type for fitness influencer")
        
        if style == 'professional':
            if params.get('clothing') and 'casual' in params.get('clothing', '').lower():
                suggestions.append("Consider business attire for professional preset")
        
        if style == 'fashion_blogger':
            if not params.get('clothing'):
                suggestions.append("Add specific clothing description for fashion blogger")
        
        # Age suggestions
        age = params.get('age', 0)
        if style == 'tech_reviewer' and age < 25:
            suggestions.append("Tech reviewers are typically portrayed as older (25+)")
        
        return suggestions
    
    def export_prompt_config(self, result: PromptResult, filename: str):
        """
        Export prompt configuration to JSON file
        Useful for debugging and sharing configs
        """
        config = result.to_dict()
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_prompt_config(self, filename: str) -> PromptResult:
        """Load prompt configuration from JSON file"""
        with open(filename, 'r') as f:
            config = json.load(f)
        
        return PromptResult(**config)
    
    def clear_cache(self):
        """Clear the prompt cache"""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "cache_size": len(self.cache),
            "cached_prompts": list(self.cache.keys())
        }


class PromptDebugger:
    """
    Debug and test prompts
    Useful during development
    """
    
    def __init__(self, manager: PromptSystemManager):
        self.manager = manager
    
    def test_all_presets(self, base_params: Dict) -> Dict[str, str]:
        """
        Test all style presets with same base params
        Returns dict of preset -> prompt
        """
        from prompt_builder import StylePreset
        
        results = {}
        
        for preset in StylePreset:
            params = base_params.copy()
            params['style_preset'] = preset.value
            
            request = GenerationRequest(
                user_id="test",
                params=params,
                num_variations=1
            )
            
            result = self.manager.generate_prompts(request)
            results[preset.value] = result.positive_prompt
        
        return results
    
    def compare_optimizations(self, params: Dict) -> Dict[str, str]:
        """
        Compare prompt with and without optimization
        """
        # Without optimization
        request_no_opt = GenerationRequest(
            user_id="test",
            params=params,
            apply_optimization=False
        )
        result_no_opt = self.manager.generate_prompts(request_no_opt)
        
        # With optimization
        request_opt = GenerationRequest(
            user_id="test",
            params=params,
            apply_optimization=True
        )
        result_opt = self.manager.generate_prompts(request_opt)
        
        return {
            "original": result_no_opt.positive_prompt,
            "optimized": result_opt.positive_prompt,
            "length_before": len(result_no_opt.positive_prompt),
            "length_after": len(result_opt.positive_prompt),
            "strength_before": result_no_opt.metadata['analysis']['estimated_strength'],
            "strength_after": result_opt.metadata['analysis']['estimated_strength']
        }
    
    def test_variations(self, params: Dict, num_variations: int = 4) -> List[str]:
        """
        Generate and return multiple variations for testing
        """
        request = GenerationRequest(
            user_id="test",
            params=params,
            num_variations=num_variations,
            variation_types=["lighting", "angle", "expression"]
        )
        
        result = self.manager.generate_prompts(request)
        return result.variations
    
    def benchmark_prompt_quality(self, params_list: List[Dict]) -> Dict:
        """
        Benchmark multiple parameter sets
        Returns quality metrics for each
        """
        results = []
        
        for i, params in enumerate(params_list):
            request = GenerationRequest(
                user_id="test",
                params=params
            )
            
            result = self.manager.generate_prompts(request)
            
            results.append({
                "index": i,
                "params": params,
                "prompt_length": len(result.positive_prompt),
                "estimated_strength": result.metadata['analysis']['estimated_strength'],
                "issues": result.metadata['analysis']['issues']
            })
        
        return {
            "results": results,
            "average_strength": sum(r['estimated_strength'] for r in results) / len(results),
            "total_tested": len(results)
        }


# Example usage demonstrating the complete system
if __name__ == "__main__":
    print("=== Complete Prompt System Demo ===\n")
    
    # Initialize system
    manager = PromptSystemManager()
    debugger = PromptDebugger(manager)
    
    # Example parameters
    example_params = {
        'age': 28,
        'gender': 'female',
        'ethnicity': 'caucasian',
        'face_shape': 'oval',
        'hair_style': 'long',
        'hair_color': 'blonde',
        'eye_color': 'blue',
        'body_type': 'athletic',
        'style_preset': 'fitness_influencer',
        'clothing': 'athletic wear',
        'expression': 'confident'
    }
    
    # 1. Validate parameters
    print("1. VALIDATING PARAMETERS")
    is_valid, errors = manager.validate_params(example_params)
    print(f"Valid: {is_valid}")
    if errors:
        print("Errors:", errors)
    print()
    
    # 2. Get suggestions
    print("2. GETTING SUGGESTIONS")
    suggestions = manager.get_prompt_suggestions(example_params)
    for suggestion in suggestions:
        print(f"  - {suggestion}")
    print()
    
    # 3. Generate prompts
    print("3. GENERATING PROMPTS")
    request = GenerationRequest(
        user_id="demo_user",
        params=example_params,
        num_variations=3,
        apply_optimization=True,
        emphasize_features=["athletic", "confident"]
    )
    
    result = manager.generate_prompts(request)
    print("Positive Prompt:")
    print(f"  {result.positive_prompt[:150]}...")
    print(f"\nNegative Prompt:")
    print(f"  {result.negative_prompt[:100]}...")
    print(f"\nVariations: {len(result.variations)}")
    print(f"Prompt Hash: {result.hash}")
    print()
    
    # 4. Analyze prompt quality
    print("4. PROMPT ANALYSIS")
    analysis = result.metadata['analysis']
    print(f"Total Tokens: {analysis['total_tokens']}")
    print(f"Estimated Strength: {analysis['estimated_strength']:.2f}")
    print(f"Issues: {analysis['issues'] if analysis['issues'] else 'None'}")
    print()
    
    # 5. Test all presets
    print("5. TESTING ALL STYLE PRESETS")
    preset_results = debugger.test_all_presets(example_params)
    for preset, prompt in preset_results.items():
        print(f"\n{preset}:")
        print(f"  {prompt[:100]}...")
    print()
    
    # 6. Compare optimization
    print("6. COMPARING OPTIMIZATION")
    comparison = debugger.compare_optimizations(example_params)
    print(f"Original length: {comparison['length_before']}")
    print(f"Optimized length: {comparison['length_after']}")
    print(f"Strength before: {comparison['strength_before']:.2f}")
    print(f"Strength after: {comparison['strength_after']:.2f}")
    print()
    
    # 7. Cache stats
    print("7. CACHE STATISTICS")
    cache_stats = manager.get_cache_stats()
    print(f"Cached prompts: {cache_stats['cache_size']}")
    print()
    
    # 8. Export configuration
    print("8. EXPORTING CONFIGURATION")
    manager.export_prompt_config(result, "example_prompt_config.json")
    print("Configuration exported to: example_prompt_config.json")
    print()
    
    print("=== Demo Complete ===")
    print("\nNext steps:")
    print("1. Integrate with SDXL generator")
    print("2. Connect to FastAPI endpoints")
    print("3. Add to your backend API")