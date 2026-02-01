
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


# =============================================================================
# PRESET VOCABULARIES
# =============================================================================

PRESET_PHOTOGRAPHY = {
    'composition': [
        "rule of thirds composition",
        "centered symmetrical composition",
        "leading lines composition",
        "framing within the frame",
        "diagonal dynamic composition",
        "minimalist negative space"
    ],
    'depth_of_field': [
        "shallow depth of field with blurred background",
        "deep focus with everything sharp",
        "selective focus on subject",
        "bokeh effect in background"
    ],
    'lighting': [
        "natural daylight",
        "golden hour warm light",
        "blue hour cool ambient light",
        "studio flash lighting",
        "dramatic side lighting",
        "soft diffused lighting",
        "harsh direct sunlight",
        "backlit silhouette"
    ],
    'mood': [
        "bright and cheerful mood",
        "dark and moody atmosphere",
        "dreamy and ethereal",
        "dramatic and intense",
        "calm and peaceful",
        "energetic and dynamic"
    ],
    'color_style': [
        "vibrant saturated colors",
        "muted pastel tones",
        "black and white monochrome",
        "warm color grading",
        "cool color grading",
        "high contrast look",
        "film-like faded colors"
    ],
    'subject_focus': [
        "portrait of a person",
        "landscape or scenery",
        "street photography scene",
        "wildlife or nature",
        "architectural details",
        "macro close-up shot",
        "action or sports moment"
    ]
}

PRESET_ART = {
    'art_style': [
        "realistic representational art",
        "impressionist brushwork style",
        "expressionist emotional distortion",
        "abstract non-representational",
        "surrealist dreamlike imagery",
        "pop art bold graphic style",
        "minimalist simple forms",
        "photorealistic detail"
    ],
    'medium': [
        "oil painting texture",
        "watercolor translucent wash",
        "acrylic bold colors",
        "pencil or charcoal drawing",
        "digital art clean lines",
        "mixed media collage",
        "pastel soft texture"
    ],
    'color_palette': [
        "warm earth tones",
        "cool blues and greens",
        "monochromatic single color",
        "complementary color contrast",
        "analogous harmonious colors",
        "vibrant rainbow spectrum",
        "muted neutral tones"
    ],
    'subject_matter': [
        "portrait or figure",
        "landscape or nature scene",
        "still life arrangement",
        "abstract shapes and forms",
        "urban or architectural",
        "mythological or fantasy",
        "everyday life scene"
    ],
    'technique': [
        "fine detailed brushwork",
        "bold expressive strokes",
        "smooth blended gradients",
        "textured impasto technique",
        "flat graphic areas",
        "gestural spontaneous marks"
    ],
    'emotional_tone': [
        "joyful and uplifting",
        "melancholic and somber",
        "mysterious and enigmatic",
        "serene and peaceful",
        "chaotic and turbulent",
        "romantic and nostalgic"
    ]
}

PRESET_PRODUCTS = {
    'product_category': [
        "electronics and technology",
        "fashion and clothing",
        "food and beverage",
        "furniture and home decor",
        "beauty and cosmetics",
        "toys and games",
        "sports and fitness equipment",
        "jewelry and accessories"
    ],
    'presentation_style': [
        "clean white background studio shot",
        "lifestyle context in use",
        "flat lay top-down arrangement",
        "hero shot dramatic angle",
        "group arrangement multiple products",
        "detail close-up shot",
        "scale comparison with object"
    ],
    'lighting_quality': [
        "soft diffused professional lighting",
        "dramatic spotlight accent",
        "natural window light",
        "ring light even illumination",
        "gradient background lighting"
    ],
    'brand_feel': [
        "premium luxury aesthetic",
        "minimalist modern clean",
        "playful colorful fun",
        "natural organic eco-friendly",
        "technical professional",
        "vintage retro nostalgic",
        "bold and edgy"
    ],
    'color_scheme': [
        "neutral white and gray",
        "brand color accent",
        "warm inviting tones",
        "cool professional tones",
        "high contrast black and white",
        "pastel soft colors"
    ],
    'surface_quality': [
        "glossy reflective surface",
        "matte smooth finish",
        "textured tactile surface",
        "metallic shiny material",
        "transparent or translucent",
        "fabric or soft material"
    ]
}


def get_preset_characteristics(preset_name: str) -> Dict[str, List[str]]:
    """
    Get a preset vocabulary by name.

    Args:
        preset_name: One of 'photography', 'art', 'products', 'default'

    Returns:
        Dictionary of characteristics for the preset

    Example:
        chars = get_preset_characteristics('photography')
    """
    presets = {
        'photography': PRESET_PHOTOGRAPHY,
        'art': PRESET_ART,
        'products': PRESET_PRODUCTS,
        'default': get_default_characteristics()
    }

    preset_name = preset_name.lower().strip()

    if preset_name not in presets:
        available = ', '.join(presets.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

    print(f"📦 {PROJECT_NAME}: Loaded preset vocabulary '{preset_name}'")
    return presets[preset_name]


def list_available_presets() -> List[str]:
    """List all available preset vocabulary names"""
    return ['photography', 'art', 'products', 'default']


def get_preset_description(preset_name: str) -> str:
    """Get a description of what a preset is designed for"""
    descriptions = {
        'photography': "Optimized for analyzing photographs: composition, lighting, mood, depth of field",
        'art': "Designed for artwork analysis: style, medium, technique, emotional tone",
        'products': "Tailored for product images: presentation, lighting, brand feel, category",
        'default': "General-purpose vocabulary for basic image analysis"
    }
    return descriptions.get(preset_name.lower(), "No description available")

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