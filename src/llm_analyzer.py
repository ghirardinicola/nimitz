#!/usr/bin/env python3
"""
NIMITZ - LLM Analyzer Module
Image analysis using multimodal LLMs (GPT-4V / Claude Vision / Gemini) instead of CLIP.

This module provides an alternative to CLIP-based analysis, using LLMs
to analyze images, generate vocabularies, and score characteristics.

Uses litellm as a unified proxy for all LLM providers.
"""

import base64
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""
    provider: str  # "openai", "anthropic", or "gemini"
    model: str     # litellm model string
    api_key: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.3


# Default configurations (litellm model format)
PROVIDER_CONFIGS = {
    "openai": LLMConfig(
        provider="openai",
        model="gpt-4o",
        api_key=os.environ.get("OPENAI_API_KEY"),
    ),
    "anthropic": LLMConfig(
        provider="anthropic",
        model="anthropic/claude-sonnet-4-20250514",
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    ),
    "gemini": LLMConfig(
        provider="gemini",
        model="gemini/gemini-2.0-flash",
        api_key=os.environ.get("GEMINI_API_KEY"),
    ),
}

# Environment variable names for each provider
API_KEY_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def get_llm_config(provider: str = "auto") -> LLMConfig:
    """
    Get LLM configuration based on provider or available API keys.

    Args:
        provider: "openai", "anthropic", "gemini", or "auto" (detect from env)

    Returns:
        LLMConfig for the selected provider
    """
    if provider == "auto":
        # Check providers in order of preference
        for prov in ["anthropic", "gemini", "openai"]:
            env_var = API_KEY_ENV_VARS[prov]
            if os.environ.get(env_var):
                return PROVIDER_CONFIGS[prov]
        raise ValueError(
            "No API key found. Set one of: ANTHROPIC_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY"
        )

    if provider not in PROVIDER_CONFIGS:
        valid = ", ".join(PROVIDER_CONFIGS.keys())
        raise ValueError(f"Unknown provider: {provider}. Use one of: {valid}, or 'auto'.")

    env_var = API_KEY_ENV_VARS[provider]
    if not os.environ.get(env_var):
        raise ValueError(f"{env_var} environment variable not set.")

    return PROVIDER_CONFIGS[provider]


def encode_image_base64(image_path: Path) -> str:
    """Encode an image to base64 string."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: Path) -> str:
    """Get media type from image extension."""
    ext = image_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return media_types.get(ext, "image/jpeg")


def call_llm(
    config: LLMConfig,
    prompt: str,
    image_path: Optional[Path] = None
) -> str:
    """
    Call LLM API using litellm as unified proxy.

    Args:
        config: LLM configuration
        prompt: Text prompt
        image_path: Optional path to image for vision models

    Returns:
        LLM response text
    """
    try:
        import litellm
    except ImportError:
        raise ImportError(
            "litellm package not installed. Run: pip install litellm"
        )

    # Build message content
    if image_path:
        base64_image = encode_image_base64(image_path)
        media_type = get_image_media_type(image_path)
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{base64_image}"
                }
            },
            {"type": "text", "text": prompt}
        ]
    else:
        content = prompt

    messages = [{"role": "user", "content": content}]

    # Call litellm
    response = litellm.completion(
        model=config.model,
        messages=messages,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )

    return response.choices[0].message.content


# =============================================================================
# IMAGE ANALYSIS
# =============================================================================

def analyze_image_llm(
    image_path: Path,
    config: Optional[LLMConfig] = None,
    language: str = "en"
) -> Dict[str, Any]:
    """
    Analyze a single image using LLM vision capabilities.

    Args:
        image_path: Path to the image file
        config: LLM configuration (auto-detected if None)
        language: Response language ("en" or "it")

    Returns:
        Dictionary with image analysis including:
        - description: Detailed description of the image
        - characteristics: Detected characteristics with scores
        - suggested_vocabulary: Suggested vocabulary based on the image
    """
    if config is None:
        config = get_llm_config("auto")

    lang_prompts = {
        "en": """Analyze this image in detail. Provide your response as valid JSON with this structure:
{
    "description": "A detailed description of the image (2-3 sentences)",
    "mood": "The overall mood/atmosphere of the image",
    "main_subject": "The main subject or focus of the image",
    "characteristics": {
        "composition": {"value": "description of composition", "score": 0-100},
        "lighting": {"value": "description of lighting", "score": 0-100},
        "color_palette": {"value": "description of colors", "score": 0-100},
        "technical_quality": {"value": "description of technical quality", "score": 0-100},
        "emotional_impact": {"value": "description of emotional impact", "score": 0-100},
        "creativity": {"value": "description of creativity/uniqueness", "score": 0-100}
    },
    "tags": ["tag1", "tag2", "tag3", "...up to 10 relevant tags"]
}

Only respond with the JSON, no other text.""",

        "it": """Analizza questa immagine in dettaglio. Fornisci la risposta come JSON valido con questa struttura:
{
    "description": "Una descrizione dettagliata dell'immagine (2-3 frasi)",
    "mood": "L'atmosfera/mood generale dell'immagine",
    "main_subject": "Il soggetto principale dell'immagine",
    "characteristics": {
        "composition": {"value": "descrizione della composizione", "score": 0-100},
        "lighting": {"value": "descrizione dell'illuminazione", "score": 0-100},
        "color_palette": {"value": "descrizione dei colori", "score": 0-100},
        "technical_quality": {"value": "descrizione della qualita tecnica", "score": 0-100},
        "emotional_impact": {"value": "descrizione dell'impatto emotivo", "score": 0-100},
        "creativity": {"value": "descrizione della creativita/unicita", "score": 0-100}
    },
    "tags": ["tag1", "tag2", "tag3", "...fino a 10 tag rilevanti"]
}

Rispondi solo con il JSON, nessun altro testo."""
    }

    prompt = lang_prompts.get(language, lang_prompts["en"])

    response = call_llm(config, prompt, image_path)

    # Parse JSON response
    try:
        # Handle potential markdown code blocks
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]

        result = json.loads(response.strip())
        result["image_path"] = str(image_path)
        result["image_name"] = image_path.name
        return result
    except json.JSONDecodeError as e:
        return {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "description": response,
            "error": f"Could not parse JSON response: {e}",
            "raw_response": response
        }


def analyze_images_batch(
    image_paths: List[Path],
    config: Optional[LLMConfig] = None,
    language: str = "en",
    progress_callback: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """
    Analyze multiple images using LLM.

    Args:
        image_paths: List of image paths to analyze
        config: LLM configuration
        language: Response language
        progress_callback: Optional callback(current, total) for progress

    Returns:
        List of analysis results for each image
    """
    if config is None:
        config = get_llm_config("auto")

    results = []
    total = len(image_paths)

    for i, image_path in enumerate(image_paths):
        try:
            result = analyze_image_llm(image_path, config, language)
            results.append(result)
        except Exception as e:
            results.append({
                "image_path": str(image_path),
                "image_name": image_path.name,
                "error": str(e)
            })

        if progress_callback:
            progress_callback(i + 1, total)

    return results


# =============================================================================
# VOCABULARY GENERATION
# =============================================================================

def generate_vocabulary_from_images(
    image_paths: List[Path],
    config: Optional[LLMConfig] = None,
    num_samples: int = 5,
    language: str = "en"
) -> Dict[str, List[str]]:
    """
    Generate a vocabulary based on sample images using LLM.

    The LLM analyzes sample images and suggests relevant characteristics
    and prompts for analyzing similar images.

    Args:
        image_paths: List of image paths to sample from
        config: LLM configuration
        num_samples: Number of images to sample for analysis
        language: Response language

    Returns:
        Dictionary of characteristics with prompt lists
    """
    if config is None:
        config = get_llm_config("auto")

    # Sample images
    import random
    samples = random.sample(image_paths, min(num_samples, len(image_paths)))

    # First, analyze sample images
    print(f"  Analyzing {len(samples)} sample images...")
    analyses = []
    for img in samples:
        try:
            analysis = analyze_image_llm(img, config, language)
            analyses.append(analysis)
        except Exception as e:
            print(f"    Warning: Could not analyze {img.name}: {e}")

    if not analyses:
        raise ValueError("Could not analyze any sample images")

    # Build prompt for vocabulary generation
    analyses_summary = "\n".join([
        f"- {a.get('description', 'No description')}"
        for a in analyses if 'description' in a
    ])

    lang_prompts = {
        "en": f"""Based on these image descriptions from a collection:

{analyses_summary}

Create a vocabulary for analyzing similar images. Respond with valid JSON containing 4-6 characteristics, each with 3-5 descriptive prompts.

Example structure:
{{
    "mood": [
        "bright and cheerful atmosphere",
        "dark and moody atmosphere",
        "calm and peaceful atmosphere",
        "dramatic and intense atmosphere"
    ],
    "composition": [
        "centered symmetrical composition",
        "rule of thirds composition",
        "dynamic diagonal composition"
    ]
}}

Create characteristics relevant to this specific image collection. Each prompt should be descriptive (5+ words).
Only respond with the JSON, no other text.""",

        "it": f"""Basandoti su queste descrizioni di immagini da una collezione:

{analyses_summary}

Crea un vocabolario per analizzare immagini simili. Rispondi con JSON valido contenente 4-6 caratteristiche, ognuna con 3-5 prompt descrittivi.

Esempio di struttura:
{{
    "atmosfera": [
        "atmosfera luminosa e allegra",
        "atmosfera scura e misteriosa",
        "atmosfera calma e serena",
        "atmosfera drammatica e intensa"
    ],
    "composizione": [
        "composizione centrata simmetrica",
        "composizione con regola dei terzi",
        "composizione diagonale dinamica"
    ]
}}

Crea caratteristiche rilevanti per questa specifica collezione. Ogni prompt deve essere descrittivo (5+ parole).
Rispondi solo con il JSON, nessun altro testo."""
    }

    prompt = lang_prompts.get(language, lang_prompts["en"])

    response = call_llm(config, prompt)

    # Parse response
    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]

        vocabulary = json.loads(response.strip())
        return vocabulary
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse vocabulary JSON: {e}\nResponse: {response}")


# =============================================================================
# SCORING WITH VOCABULARY
# =============================================================================

def score_image_with_vocabulary(
    image_path: Path,
    vocabulary: Dict[str, List[str]],
    config: Optional[LLMConfig] = None,
    language: str = "en"
) -> Dict[str, Any]:
    """
    Score an image against a specific vocabulary using LLM.

    Args:
        image_path: Path to the image
        vocabulary: Vocabulary dictionary with characteristics and prompts
        config: LLM configuration
        language: Response language

    Returns:
        Dictionary with scores for each characteristic
    """
    if config is None:
        config = get_llm_config("auto")

    # Build vocabulary description for prompt
    vocab_description = json.dumps(vocabulary, indent=2, ensure_ascii=False)

    lang_prompts = {
        "en": f"""Analyze this image using the following vocabulary. For each characteristic, score how well each prompt describes the image (0-100).

Vocabulary:
{vocab_description}

Respond with valid JSON in this format:
{{
    "characteristic_name": {{
        "scores": {{"prompt1": score, "prompt2": score, ...}},
        "best_match": "the prompt that best describes this aspect",
        "best_score": highest_score
    }},
    ...
}}

Be precise with scores:
- 90-100: Perfect match
- 70-89: Strong match
- 50-69: Moderate match
- 30-49: Weak match
- 0-29: No match

Only respond with the JSON, no other text.""",

        "it": f"""Analizza questa immagine usando il seguente vocabolario. Per ogni caratteristica, dai un punteggio (0-100) a quanto ogni prompt descrive l'immagine.

Vocabolario:
{vocab_description}

Rispondi con JSON valido in questo formato:
{{
    "nome_caratteristica": {{
        "scores": {{"prompt1": punteggio, "prompt2": punteggio, ...}},
        "best_match": "il prompt che meglio descrive questo aspetto",
        "best_score": punteggio_migliore
    }},
    ...
}}

Sii preciso con i punteggi:
- 90-100: Corrispondenza perfetta
- 70-89: Corrispondenza forte
- 50-69: Corrispondenza moderata
- 30-49: Corrispondenza debole
- 0-29: Nessuna corrispondenza

Rispondi solo con il JSON, nessun altro testo."""
    }

    prompt = lang_prompts.get(language, lang_prompts["en"])

    response = call_llm(config, prompt, image_path)

    try:
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]

        scores = json.loads(response.strip())
        return {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "characteristics": scores
        }
    except json.JSONDecodeError as e:
        return {
            "image_path": str(image_path),
            "image_name": image_path.name,
            "error": f"Could not parse scores JSON: {e}",
            "raw_response": response
        }


def generate_card_data_llm(
    image_path: Path,
    vocabulary: Dict[str, List[str]],
    config: Optional[LLMConfig] = None,
    language: str = "en"
) -> Dict[str, Any]:
    """
    Generate card data for an image using LLM analysis.

    This produces data compatible with the existing image_card module.

    Args:
        image_path: Path to the image
        vocabulary: Vocabulary to use for scoring
        config: LLM configuration
        language: Response language

    Returns:
        Card data dictionary compatible with image_card functions
    """
    if config is None:
        config = get_llm_config("auto")

    # Get scores
    scored = score_image_with_vocabulary(image_path, vocabulary, config, language)

    if "error" in scored:
        return scored

    # Transform to card format
    characteristics = {}
    dominant_features = []

    for char_name, char_data in scored.get("characteristics", {}).items():
        if isinstance(char_data, dict):
            best_score = char_data.get("best_score", 0)
            best_prompt = char_data.get("best_match", "")

            # Normalize score to 0-1 range
            normalized_score = best_score / 100.0

            characteristics[char_name] = {
                "max_score": normalized_score,
                "max_prompt": best_prompt,
                "mean_score": normalized_score,
                "scores": char_data.get("scores", {})
            }

            confidence = "high" if best_score >= 70 else "medium" if best_score >= 50 else "low"

            dominant_features.append({
                "characteristic": char_name,
                "prompt": best_prompt,
                "score": normalized_score,
                "confidence": confidence
            })

    # Sort by score
    dominant_features.sort(key=lambda x: x["score"], reverse=True)

    # Calculate summary
    all_scores = [f["score"] for f in dominant_features]

    return {
        "image_path": str(image_path),
        "image_name": image_path.name,
        "characteristics": characteristics,
        "dominant_features": dominant_features[:6],  # Top 6
        "feature_summary": {
            "overall_max": max(all_scores) if all_scores else 0,
            "overall_mean": sum(all_scores) / len(all_scores) if all_scores else 0,
            "high_confidence_features": sum(1 for f in dominant_features if f["confidence"] == "high"),
            "medium_confidence_features": sum(1 for f in dominant_features if f["confidence"] == "medium"),
        },
        "analysis_method": "llm"
    }


def generate_cards_batch_llm(
    image_paths: List[Path],
    vocabulary: Dict[str, List[str]],
    config: Optional[LLMConfig] = None,
    language: str = "en",
    progress_callback: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """
    Generate card data for multiple images using LLM.

    Args:
        image_paths: List of image paths
        vocabulary: Vocabulary for scoring
        config: LLM configuration
        language: Response language
        progress_callback: Optional callback(current, total)

    Returns:
        List of card data dictionaries
    """
    if config is None:
        config = get_llm_config("auto")

    cards = []
    total = len(image_paths)

    for i, image_path in enumerate(image_paths):
        try:
            card = generate_card_data_llm(image_path, vocabulary, config, language)
            cards.append(card)
        except Exception as e:
            cards.append({
                "image_path": str(image_path),
                "image_name": image_path.name,
                "error": str(e)
            })

        if progress_callback:
            progress_callback(i + 1, total)

    return cards


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_llm_availability() -> Dict[str, bool]:
    """Check which LLM providers are available."""
    return {
        "openai": bool(os.environ.get("OPENAI_API_KEY")),
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "gemini": bool(os.environ.get("GEMINI_API_KEY")),
    }


def get_available_provider() -> Optional[str]:
    """Get the first available LLM provider."""
    availability = check_llm_availability()
    for provider in ["anthropic", "gemini", "openai"]:
        if availability[provider]:
            return provider
    return None


def print_llm_status():
    """Print LLM availability status."""
    availability = check_llm_availability()
    print("\n LLM Provider Status")
    print("=" * 40)
    for provider, available in availability.items():
        status = "Available" if available else "Not configured"
        icon = "" if available else ""
        env_var = API_KEY_ENV_VARS.get(provider, "")
        print(f"  {icon} {provider.capitalize():10} {status:15} ({env_var})")
    print()
    print("  Install litellm: pip install litellm")
    print()
