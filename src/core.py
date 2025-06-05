
#!/usr/bin/env python3
"""
NIMITZ - Core Module
Semantic Image Clustering with Multi-Characteristic Analysis
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

PROJECT_NAME = "NIMITZ"

def load_characteristics_from_json(json_path: str) -> Dict[str, List[str]]:
    """Load characteristics from JSON file"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        characteristics = data.get('characteristics', data)
        print(f"📁 {PROJECT_NAME}: Loaded characteristics from {json_path}")
        return characteristics
    except Exception as e:
        print(f"❌ Error loading from {json_path}: {e}")
        return {}

def get_default_characteristics() -> Dict[str, List[str]]:
    """Get default set of characteristics"""
    return {
        'color_temperature': [
            "warm color palette with reds, oranges, and yellows",
            "cool color palette with blues, greens, and purples", 
            "neutral color palette with grays, beiges, and whites"
        ],
        'color_saturation': [
            "vibrant and saturated colors",
            "muted and desaturated colors",
            "monochromatic color scheme"
        ],
        'lighting_time': [
            "early morning sunrise lighting",
            "bright midday sunlight",
            "afternoon golden hour lighting",
            "evening sunset lighting",
            "nighttime artificial lighting"
        ],
        'lighting_quality': [
            "soft and diffused lighting",
            "harsh and direct lighting",
            "dramatic lighting with strong shadows",
            "even and balanced lighting"
        ]
    }

def load_images(image_directory: str) -> List[Path]:
    """Load image paths from directory"""
    image_dir = Path(image_directory)
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    image_paths = []
    for ext in supported_formats:
        image_paths.extend(list(image_dir.glob(f"*{ext}")))
        image_paths.extend(list(image_dir.glob(f"*{ext.upper()}")))
        
    print(f"🖼️  {PROJECT_NAME}: Found {len(image_paths)} images to analyze")
    return image_paths

def validate_characteristics(characteristics: Dict[str, List[str]]) -> bool:
    """Validate characteristics dictionary"""
    if not isinstance(characteristics, dict):
        return False
    
    for name, prompts in characteristics.items():
        if not isinstance(prompts, list) or len(prompts) == 0:
            print(f"⚠️  Invalid prompts for characteristic '{name}'")
            return False
    
    return True

def print_characteristics_summary(characteristics: Dict[str, List[str]]) -> None:
    """Print summary of available characteristics"""
    print(f"\n🗂️  {PROJECT_NAME} - CHARACTERISTICS ARSENAL")
    print("=" * 50)
    
    for name, prompts in characteristics.items():
        print(f"\n🎯 {name.upper().replace('_', ' ')}")
        for i, prompt in enumerate(prompts, 1):
            print(f"   {i}. {prompt}")
            
    total_prompts = sum(len(p) for p in characteristics.values())
    print(f"\n⚓ Total: {len(characteristics)} categories")
    print(f"💪 Total prompts: {total_prompts}")

def get_characteristics_summary(characteristics: Dict[str, List[str]]) -> Dict:
    """Get characteristics summary as dictionary"""
    return {
        'total_categories': len(characteristics),
        'total_prompts': sum(len(prompts) for prompts in characteristics.values()),
        'categories': list(characteristics.keys()),
        'prompts_per_category': {name: len(prompts) for name, prompts in characteristics.items()}
    }

def save_results(results_data: Dict, output_dir: str = "../results") -> Path:
    """Save analysis results to files"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"💾 {PROJECT_NAME}: Saving results to {output_path}")
    
    # Save different components based on what's provided
    for filename, data in results_data.items():
        filepath = output_path / filename
        
        if filename.endswith('.json'):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        elif filename.endswith('.csv') and hasattr(data, 'to_csv'):
            data.to_csv(filepath, index=False)
    
    print(f"📁 {PROJECT_NAME}: Results saved to {output_path}")
    return output_path

def save_characteristics_config(characteristics: Dict[str, List[str]], filename: str) -> None:
    """Save characteristics to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump({"characteristics": characteristics}, f, indent=2, ensure_ascii=False)
    print(f"💾 Characteristics saved to {filename}")