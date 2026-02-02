#!/usr/bin/env python3
"""
NIMITZ - Command Line Interface
Usage-friendly CLI for image analysis and card generation
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

from core import (
    load_images,
    get_preset_characteristics,
    get_default_characteristics,
    list_available_presets,
    get_preset_description,
    validate_characteristics,
)
from embed import (
    initialize_clip_model,
    extract_image_features,
    extract_text_features,
    compute_similarity_matrices,
)
from image_card import (
    generate_image_cards_data,
    print_simple_image_cards,
    create_visual_image_cards,
    export_cards_to_csv,
)
from vocabulary_wizard import (
    run_wizard,
    quick_validate,
    analyze_prompt_quality,
)
from llm_analyzer import (
    get_llm_config,
    analyze_image_llm,
    analyze_images_batch,
    generate_vocabulary_from_images,
    generate_card_data_llm,
    generate_cards_batch_llm,
    check_llm_availability,
    print_llm_status,
)
from gaming import (
    compare_cards,
    battle_cards,
    calculate_collection_stats,
    enhance_cards_with_gaming_stats,
    print_battle_result,
    print_comparison_result,
    print_collection_stats,
    RarityTier,
    RARITY_SYMBOLS,
)
from deck import (
    Deck,
    create_deck_from_cards,
    merge_decks,
    print_deck_info,
    list_cards_in_deck,
)
from pdf_export import (
    check_pdf_support,
    export_cards_to_pdf,
    export_single_card_pdf,
    export_cards_to_png,
    print_pdf_export_info,
)
from image_retrieval import (
    retrieve_image,
    batch_retrieve_images,
    check_api_credentials,
    create_placeholder_image,
    ImageRetrievalError,
)


def describe_single_image(
    image_path: str,
    preset: str = "photography",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Analyze and describe a single image.

    Args:
        image_path: Path to the image file
        preset: Vocabulary preset to use
        verbose: Show detailed output

    Returns:
        Dictionary with image analysis results
    """
    image_path = Path(image_path)

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    if not image_path.is_file():
        print(f"Error: Not a file: {image_path}")
        sys.exit(1)

    print(f"\n Analyzing: {image_path.name}")
    print("-" * 50)

    # Get characteristics
    characteristics = get_preset_characteristics(preset)

    # Initialize model
    model, preprocess, device = initialize_clip_model()

    # Extract features for single image
    image_features, valid_paths = extract_image_features(
        [image_path], model, preprocess, device, batch_size=1
    )

    if not valid_paths:
        print(f"Error: Could not process image: {image_path}")
        sys.exit(1)

    # Extract text features
    characteristic_features, characteristic_labels = extract_text_features(
        characteristics, model, device
    )

    # Compute similarities
    similarity_matrices = compute_similarity_matrices(
        image_features, characteristic_features, characteristic_labels, valid_paths
    )

    # Generate card data
    cards_data = generate_image_cards_data(
        valid_paths, similarity_matrices, characteristic_labels
    )

    if not cards_data:
        print("Error: Could not generate card data")
        sys.exit(1)

    card = cards_data[0]

    # Print results
    print(f"\n IMAGE: {card['image_name']}")
    print("=" * 50)

    # Show dominant features
    if card['dominant_features']:
        print("\n TOP FEATURES:")
        for i, feature in enumerate(card['dominant_features'], 1):
            score_bar = create_score_bar(feature['score'])
            confidence = "HIGH" if feature['confidence'] == 'high' else "MED"
            print(f"  {i}. [{confidence}] {feature['characteristic']}: {feature['prompt']}")
            print(f"     {score_bar} {feature['score']:.2f}")

    # Show characteristic breakdown
    if verbose:
        print("\n DETAILED ANALYSIS:")
        for char_name, char_info in card['characteristics'].items():
            print(f"\n  {char_name.upper().replace('_', ' ')}:")
            print(f"    Best: {char_info['max_prompt']} ({char_info['max_score']:.2f})")
            print(f"    Avg:  {char_info['mean_score']:.2f}")

    # Summary
    summary = card['feature_summary']
    print(f"\n SUMMARY:")
    print(f"  Overall score: {summary['overall_max']:.2f}")
    print(f"  High confidence features: {summary['high_confidence_features']}")
    print(f"  Medium confidence: {summary['medium_confidence_features']}")

    return card


def analyze_directory(
    directory: str,
    preset: str = "photography",
    output_dir: Optional[str] = None,
    n_clusters: int = 5,
    no_visual: bool = False,
    quiet: bool = False
) -> Dict[str, Any]:
    """
    Analyze all images in a directory.

    Args:
        directory: Path to directory with images
        preset: Vocabulary preset to use
        output_dir: Output directory (default: ./nimitz_output)
        n_clusters: Number of clusters for grouping
        no_visual: Skip visual card generation
        quiet: Minimal output

    Returns:
        Dictionary with analysis results
    """
    directory = Path(directory)

    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        sys.exit(1)

    if not directory.is_dir():
        print(f"Error: Not a directory: {directory}")
        sys.exit(1)

    # Setup output directory
    if output_dir is None:
        output_dir = directory / "nimitz_output"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)

    if not quiet:
        print(f"\n NIMITZ Analysis")
        print("=" * 50)
        print(f"  Directory: {directory}")
        print(f"  Preset: {preset}")
        print(f"  Output: {output_dir}")
        print()

    # Get characteristics
    characteristics = get_preset_characteristics(preset)

    # Load images
    image_paths = load_images(str(directory))

    if not image_paths:
        print(f"Error: No images found in {directory}")
        sys.exit(1)

    if not quiet:
        print(f"  Found {len(image_paths)} images")

    # Initialize model
    model, preprocess, device = initialize_clip_model()

    # Extract features
    image_features, valid_paths = extract_image_features(
        image_paths, model, preprocess, device
    )

    characteristic_features, characteristic_labels = extract_text_features(
        characteristics, model, device
    )

    # Compute similarities
    similarity_matrices = compute_similarity_matrices(
        image_features, characteristic_features, characteristic_labels, valid_paths
    )

    # Generate cards
    cards_data = generate_image_cards_data(
        valid_paths, similarity_matrices, characteristic_labels
    )

    # Print summary cards
    if not quiet:
        print("\n" + "=" * 50)
        print(" IMAGE CARDS")
        print("=" * 50)
        print_simple_image_cards(cards_data, max_cards=5 if len(cards_data) > 5 else None)

    # Export CSV
    csv_path = output_dir / "cards.csv"
    export_cards_to_csv(cards_data, str(csv_path))

    # Create visual cards
    if not no_visual:
        visual_dir = output_dir / "cards"
        create_visual_image_cards(cards_data, str(visual_dir))

    # Final summary
    if not quiet:
        print("\n" + "=" * 50)
        print(" ANALYSIS COMPLETE")
        print("=" * 50)
        print(f"  Images analyzed: {len(valid_paths)}")
        print(f"  Cards generated: {len(cards_data)}")
        print(f"\n  Output files:")
        print(f"    CSV: {csv_path}")
        if not no_visual:
            print(f"    Visual cards: {visual_dir}")

    return {
        'cards_data': cards_data,
        'valid_paths': valid_paths,
        'output_dir': output_dir
    }


def create_score_bar(score: float, width: int = 20) -> str:
    """Create a visual score bar."""
    filled = int(score * width)
    empty = width - filled
    return "[" + "#" * filled + "-" * empty + "]"


def list_presets():
    """List all available presets with descriptions."""
    print("\n Available Presets")
    print("=" * 50)

    for preset_name in list_available_presets():
        description = get_preset_description(preset_name)
        print(f"\n  {preset_name}")
        print(f"    {description}")

    print()


def run_vocabulary_wizard(
    image_directory: Optional[str] = None,
    output_file: Optional[str] = None,
    analyze_after: bool = False
):
    """
    Run the interactive vocabulary wizard.

    Args:
        image_directory: Directory with sample images for suggestions
        output_file: Output file to save vocabulary
        analyze_after: Run analysis after creating vocabulary
    """
    print("\n NIMITZ - Vocabulary Wizard")
    print("=" * 50)

    # Run wizard
    vocabulary = run_wizard(image_directory)

    if not vocabulary:
        print("\nVocabulary creation cancelled or empty.")
        return

    # Save if output specified
    if output_file:
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"characteristics": vocabulary}, f, indent=2, ensure_ascii=False)
        print(f"\n Vocabulary saved to: {output_file}")

    # Analyze if requested
    if analyze_after and image_directory:
        print("\n Running analysis with new vocabulary...")
        from main import run_nimitz_pipeline
        run_nimitz_pipeline(
            image_directory=image_directory,
            characteristics=vocabulary,
            visualize=True
        )


def validate_vocabulary_file(vocabulary_file: str):
    """
    Validate a vocabulary JSON file.

    Args:
        vocabulary_file: Path to JSON file to validate
    """
    import json
    from pathlib import Path

    filepath = Path(vocabulary_file)

    if not filepath.exists():
        print(f"Error: File not found: {vocabulary_file}")
        sys.exit(1)

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}")
        sys.exit(1)

    # Extract characteristics
    characteristics = data.get('characteristics', data)

    if not isinstance(characteristics, dict):
        print("Error: Invalid vocabulary format. Expected 'characteristics' dictionary.")
        sys.exit(1)

    # Validate
    report = quick_validate(characteristics)

    print("\n VOCABULARY VALIDATION REPORT")
    print("=" * 50)
    print(f"  File: {vocabulary_file}")
    print(f"  Characteristics: {report['total_characteristics']}")
    print(f"  Total prompts: {report['total_prompts']}")
    print()

    # Per-characteristic quality
    print(" Quality per characteristic:")
    for name, quality in report['characteristic_quality'].items():
        indicator = "" if quality['score'] >= 70 else "" if quality['score'] >= 50 else ""
        print(f"  {indicator} {name}: {quality['score']}/100 ({quality['count']} prompts)")
        if quality['issues']:
            for issue in quality['issues'][:2]:
                print(f"      - {issue}")

    # Overall assessment
    print()
    if report['valid']:
        print(" Vocabulary is valid and ready to use!")
    else:
        print(" Vocabulary has quality issues:")
        for issue in report['issues']:
            print(f"    - {issue}")


# =============================================================================
# LLM-BASED FUNCTIONS
# =============================================================================

def describe_single_image_llm(
    image_path: str,
    provider: str = "auto",
    verbose: bool = False,
    language: str = "en"
) -> Dict[str, Any]:
    """
    Analyze a single image using LLM vision capabilities.

    Args:
        image_path: Path to the image file
        provider: LLM provider (openai, anthropic, auto)
        verbose: Show detailed output
        language: Response language (en/it)

    Returns:
        Dictionary with image analysis results
    """
    image_path = Path(image_path)

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    print(f"\n Analyzing with LLM: {image_path.name}")
    print("-" * 50)

    try:
        config = get_llm_config(provider)
        print(f"  Using: {config.provider.capitalize()} ({config.model})")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    result = analyze_image_llm(image_path, config, language)

    if "error" in result:
        print(f"Error: {result['error']}")
        return result

    # Print results
    print(f"\n IMAGE: {result['image_name']}")
    print("=" * 50)

    if "description" in result:
        print(f"\n DESCRIPTION:")
        print(f"  {result['description']}")

    if "mood" in result:
        print(f"\n MOOD: {result['mood']}")

    if "main_subject" in result:
        print(f"\n SUBJECT: {result['main_subject']}")

    if "characteristics" in result:
        print(f"\n CHARACTERISTICS:")
        for char_name, char_data in result["characteristics"].items():
            if isinstance(char_data, dict):
                score = char_data.get("score", 0)
                value = char_data.get("value", "")
                score_bar = create_score_bar(score / 100)
                print(f"  {char_name.replace('_', ' ').title()}:")
                print(f"    {score_bar} {score}")
                if verbose:
                    print(f"    {value}")

    if "tags" in result and result["tags"]:
        print(f"\n TAGS: {', '.join(result['tags'])}")

    return result


def analyze_directory_llm(
    directory: str,
    preset: str = "photography",
    output_dir: Optional[str] = None,
    provider: str = "auto",
    no_visual: bool = False,
    quiet: bool = False,
    language: str = "en"
) -> Dict[str, Any]:
    """
    Analyze all images in a directory using LLM.

    Args:
        directory: Path to directory with images
        preset: Vocabulary preset to use
        output_dir: Output directory
        provider: LLM provider
        no_visual: Skip visual card generation
        quiet: Minimal output
        language: Response language

    Returns:
        Dictionary with analysis results
    """
    directory = Path(directory)

    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        sys.exit(1)

    # Setup output directory
    if output_dir is None:
        output_dir = directory / "nimitz_output_llm"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)

    try:
        config = get_llm_config(provider)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not quiet:
        print(f"\n NIMITZ LLM Analysis")
        print("=" * 50)
        print(f"  Directory: {directory}")
        print(f"  Provider: {config.provider.capitalize()} ({config.model})")
        print(f"  Preset: {preset}")
        print(f"  Output: {output_dir}")
        print()

    # Get characteristics
    characteristics = get_preset_characteristics(preset)

    # Load images
    image_paths = load_images(str(directory))

    if not image_paths:
        print(f"Error: No images found in {directory}")
        sys.exit(1)

    if not quiet:
        print(f"  Found {len(image_paths)} images")
        print(f"  Analyzing with LLM (this may take a while)...")

    # Generate cards using LLM
    def progress(current, total):
        if not quiet:
            print(f"    Processing {current}/{total}...", end="\r")

    cards_data = generate_cards_batch_llm(
        image_paths, characteristics, config, language, progress
    )

    if not quiet:
        print()

    # Filter successful cards
    valid_cards = [c for c in cards_data if "error" not in c]

    if not quiet:
        print(f"\n Successfully analyzed: {len(valid_cards)}/{len(image_paths)} images")

    # Print summary cards
    if not quiet and valid_cards:
        print("\n" + "=" * 50)
        print(" IMAGE CARDS (LLM)")
        print("=" * 50)
        for card in valid_cards[:5]:
            print(f"\n  {card['image_name']}")
            for feature in card.get('dominant_features', [])[:3]:
                score_bar = create_score_bar(feature['score'])
                print(f"    {feature['characteristic']}: {score_bar} {feature['score']:.2f}")

    # Export to JSON
    import json
    json_path = output_dir / "cards_llm.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(cards_data, f, indent=2, ensure_ascii=False)

    # Export CSV
    csv_path = output_dir / "cards_llm.csv"
    export_cards_to_csv(valid_cards, str(csv_path))

    # Create visual cards
    if not no_visual and valid_cards:
        visual_dir = output_dir / "cards"
        create_visual_image_cards(valid_cards, str(visual_dir))

    # Final summary
    if not quiet:
        print("\n" + "=" * 50)
        print(" ANALYSIS COMPLETE (LLM)")
        print("=" * 50)
        print(f"  Images analyzed: {len(valid_cards)}")
        print(f"  Cards generated: {len(valid_cards)}")
        print(f"\n  Output files:")
        print(f"    JSON: {json_path}")
        print(f"    CSV: {csv_path}")
        if not no_visual:
            print(f"    Visual cards: {visual_dir}")

    return {
        'cards_data': cards_data,
        'valid_cards': valid_cards,
        'output_dir': output_dir
    }


def generate_vocabulary_llm(
    directory: str,
    output_file: Optional[str] = None,
    provider: str = "auto",
    num_samples: int = 5,
    language: str = "en"
):
    """
    Generate a vocabulary using LLM analysis of sample images.

    Args:
        directory: Directory with sample images
        output_file: Output file for vocabulary (JSON)
        provider: LLM provider
        num_samples: Number of images to sample
        language: Response language
    """
    directory = Path(directory)

    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        sys.exit(1)

    try:
        config = get_llm_config(provider)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"\n NIMITZ - LLM Vocabulary Generator")
    print("=" * 50)
    print(f"  Provider: {config.provider.capitalize()} ({config.model})")
    print(f"  Directory: {directory}")
    print()

    # Load images
    image_paths = load_images(str(directory))

    if not image_paths:
        print(f"Error: No images found in {directory}")
        sys.exit(1)

    print(f"  Found {len(image_paths)} images, sampling {min(num_samples, len(image_paths))}...")

    try:
        vocabulary = generate_vocabulary_from_images(
            image_paths, config, num_samples, language
        )
    except Exception as e:
        print(f"Error generating vocabulary: {e}")
        sys.exit(1)

    # Print vocabulary
    print(f"\n GENERATED VOCABULARY:")
    print("=" * 50)

    for char_name, prompts in vocabulary.items():
        print(f"\n  {char_name.upper().replace('_', ' ')}")
        for prompt in prompts:
            print(f"    - {prompt}")

    total_prompts = sum(len(p) for p in vocabulary.values())
    print(f"\n  Total: {len(vocabulary)} characteristics, {total_prompts} prompts")

    # Save if output specified
    if output_file:
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"characteristics": vocabulary}, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved to: {output_file}")

    return vocabulary


# =============================================================================
# GAMING FUNCTIONS
# =============================================================================

def load_cards_from_json(filepath: str) -> List[Dict[str, Any]]:
    """Load cards from a JSON file (from previous analysis)"""
    import json
    path = Path(filepath)

    if not path.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle both formats: list of cards or dict with cards_data
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'cards_data' in data:
        return data['cards_data']
    elif isinstance(data, dict) and 'cards' in data:
        return data['cards']
    else:
        print("Error: Invalid cards file format")
        sys.exit(1)


def compare_cards_command(
    cards_file: str,
    card1_name: str,
    card2_name: str,
    category: Optional[str] = None
):
    """Compare two cards from a cards file"""
    cards = load_cards_from_json(cards_file)

    # Find the cards
    card1 = None
    card2 = None

    for card in cards:
        name = card.get('image_name', '')
        if card1_name.lower() in name.lower():
            card1 = card
        if card2_name.lower() in name.lower():
            card2 = card

    if card1 is None:
        print(f"Error: Card '{card1_name}' not found")
        available = [c.get('image_name', '') for c in cards[:10]]
        print(f"Available cards (first 10): {', '.join(available)}")
        sys.exit(1)

    if card2 is None:
        print(f"Error: Card '{card2_name}' not found")
        sys.exit(1)

    result = compare_cards(card1, card2, category)
    print_comparison_result(result)


def battle_cards_command(
    cards_file: str,
    card1_name: str,
    card2_name: str,
    rounds: int = 5
):
    """Battle between two cards"""
    cards = load_cards_from_json(cards_file)

    # Find the cards
    card1 = None
    card2 = None

    for card in cards:
        name = card.get('image_name', '')
        if card1_name.lower() in name.lower():
            card1 = card
        if card2_name.lower() in name.lower():
            card2 = card

    if card1 is None:
        print(f"Error: Card '{card1_name}' not found")
        sys.exit(1)

    if card2 is None:
        print(f"Error: Card '{card2_name}' not found")
        sys.exit(1)

    result = battle_cards(card1, card2, rounds)
    print_battle_result(result)


def show_rarity_command(cards_file: str, top_n: int = 10):
    """Show rarity information for cards"""
    cards = load_cards_from_json(cards_file)
    enhanced = enhance_cards_with_gaming_stats(cards)

    print("\n CARD RARITY RANKINGS")
    print("=" * 60)

    # Sort by rarity
    sorted_cards = sorted(
        enhanced,
        key=lambda c: c.get('rarity_score', 0),
        reverse=True
    )

    for i, card in enumerate(sorted_cards[:top_n], 1):
        name = card.get('image_name', 'Unknown')
        rarity = card.get('rarity_score', 0)
        tier = card.get('rarity_tier', 'common')
        symbol = card.get('rarity_symbol', '')
        power = card.get('power_level', 0)

        print(f"\n  {i}. {symbol} {name}")
        print(f"     Rarity: {rarity:.1f} ({tier.title()})")
        print(f"     Power:  {power:.1f}")

    print()


def show_stats_command(cards_file: str):
    """Show collection statistics"""
    cards = load_cards_from_json(cards_file)
    stats = calculate_collection_stats(cards)
    print_collection_stats(stats)


def deck_create_command(
    cards_file: str,
    deck_name: str,
    output_file: str,
    card_names: Optional[List[str]] = None,
    top_n: Optional[int] = None
):
    """Create a new deck from cards"""
    cards = load_cards_from_json(cards_file)

    if card_names:
        # Select specific cards
        selected = []
        for name in card_names:
            for card in cards:
                if name.lower() in card.get('image_name', '').lower():
                    selected.append(card)
                    break
        cards = selected
    elif top_n:
        # Select top N by power
        enhanced = enhance_cards_with_gaming_stats(cards)
        cards = sorted(
            enhanced,
            key=lambda c: c.get('power_level', 0),
            reverse=True
        )[:top_n]

    deck = create_deck_from_cards(cards, name=deck_name)
    saved_path = deck.save(output_file)

    print(f"\n Deck created: {deck_name}")
    print(f"   Cards: {deck.size()}")
    print(f"   Saved to: {saved_path}")
    print()


def deck_show_command(deck_file: str, list_cards: bool = False):
    """Show deck information"""
    deck = Deck.load(deck_file)

    if list_cards:
        list_cards_in_deck(deck, show_stats=True)
    else:
        print_deck_info(deck)


def deck_add_command(deck_file: str, cards_file: str, card_names: List[str]):
    """Add cards to a deck"""
    deck = Deck.load(deck_file)
    cards = load_cards_from_json(cards_file)

    added = 0
    for name in card_names:
        for card in cards:
            if name.lower() in card.get('image_name', '').lower():
                if deck.add_card(card):
                    added += 1
                    print(f"  Added: {card.get('image_name')}")
                else:
                    print(f"  Skipped (duplicate): {card.get('image_name')}")
                break

    deck.save(deck_file)
    print(f"\n Added {added} cards to deck")
    print(f" Deck now has {deck.size()} cards")


def deck_remove_command(deck_file: str, card_names: List[str]):
    """Remove cards from a deck"""
    deck = Deck.load(deck_file)

    removed = 0
    for name in card_names:
        if deck.remove_card(name):
            removed += 1
            print(f"  Removed: {name}")
        else:
            # Try partial match
            for card in deck.cards:
                if name.lower() in card.get('image_name', '').lower():
                    if deck.remove_card(card.get('image_name')):
                        removed += 1
                        print(f"  Removed: {card.get('image_name')}")
                    break

    deck.save(deck_file)
    print(f"\n Removed {removed} cards from deck")
    print(f" Deck now has {deck.size()} cards")


def export_pdf_command(
    cards_file: str,
    output_file: str,
    page_size: str = "A4"
):
    """Export cards to PDF"""
    if not check_pdf_support():
        print("Error: PDF export requires reportlab")
        print("Install with: pip install reportlab")
        sys.exit(1)

    cards = load_cards_from_json(cards_file)
    export_cards_to_pdf(cards, output_file, page_size=page_size)


def export_png_command(
    cards_file: str,
    output_dir: str
):
    """Export cards as PNG files"""
    cards = load_cards_from_json(cards_file)
    export_cards_to_png(cards, output_dir)


def retrieve_status_command():
    """Show image retrieval API status"""
    import os

    print("\n Image Retrieval API Status")
    print("=" * 50)

    sources = check_api_credentials()

    for source, available in sources.items():
        status = "✓ Available" if available else "✗ Not configured"
        print(f"\n{source.capitalize()}: {status}")

        if not available:
            if source == 'unsplash':
                print("  Set UNSPLASH_ACCESS_KEY environment variable")
                print("  Get key at: https://unsplash.com/developers")
            elif source == 'pexels':
                print("  Set PEXELS_API_KEY environment variable")
                print("  Get key at: https://www.pexels.com/api/")

    print("\n" + "=" * 50)

    if not any(sources.values()):
        print("\n⚠ No image sources configured!")
        print("Configure at least one API key to use image retrieval.")
        sys.exit(1)


def retrieve_single_command(
    description: str,
    output_dir: str = './nimitz_output',
    preset: str = 'photography',
    source: str = 'unsplash',
    cache_dir: Optional[str] = None,
    use_clip: bool = True,
    analyze: bool = True
):
    """Retrieve a single image and optionally analyze it"""
    import os
    import json

    # Check API credentials
    sources = check_api_credentials()
    if not sources.get(source.lower()):
        print(f"\n✗ Error: {source} API not configured")
        print(f"Set {source.upper()}_ACCESS_KEY or {source.upper()}_API_KEY environment variable")
        sys.exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set up cache
    if cache_dir is None:
        cache_dir = os.path.expanduser('~/.nimitz_cache')

    print(f"\n Retrieving image for: {description}")
    print(f" Source: {source}")
    print(f" Output: {output_dir}")

    # Initialize CLIP if needed for selection
    model, device = None, None
    characteristics = None

    if use_clip:
        try:
            characteristics = get_preset_characteristics(preset)
            model, device = initialize_clip_model()
            print(" Using CLIP for image selection")
        except Exception as e:
            print(f" Warning: CLIP not available ({e}), using first result")
            use_clip = False

    # Retrieve image
    try:
        image_path, metadata = retrieve_image(
            description=description,
            output_dir=output_dir,
            source_name=source,
            cache_dir=cache_dir,
            characteristics=characteristics,
            model=model,
            device=device,
            num_candidates=5
        )

        print(f"\n✓ Image retrieved successfully!")
        print(f"   Path: {image_path}")
        print(f"   Author: {metadata['author']}")
        print(f"   License: {metadata['license']}")
        print(f"   Attribution: {metadata['attribution']}")

        # Save metadata
        metadata_path = os.path.join(output_dir, 'metadata.json')
        metadata_list = []
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata_list = json.load(f)

        metadata_list.append({
            'description': description,
            'image_path': image_path,
            'metadata': metadata
        })

        with open(metadata_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)

        # Optionally analyze the image
        if analyze:
            print(f"\n Analyzing retrieved image...")
            describe_single_image(image_path, preset=preset, verbose=True)

    except ImageRetrievalError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


def retrieve_batch_command(
    descriptions_file: str,
    output_dir: str = './nimitz_output',
    preset: str = 'photography',
    source: str = 'unsplash',
    cache_dir: Optional[str] = None,
    use_clip: bool = True,
    analyze: bool = True,
    use_placeholder: bool = True
):
    """Retrieve multiple images and generate cards"""
    import os
    import json
    from tqdm import tqdm

    # Check API credentials
    sources = check_api_credentials()
    if not sources.get(source.lower()):
        print(f"\n✗ Error: {source} API not configured")
        print(f"Set {source.upper()}_ACCESS_KEY or {source.upper()}_API_KEY environment variable")
        sys.exit(1)

    # Load descriptions
    descriptions = []
    descriptions_path = Path(descriptions_file)

    if not descriptions_path.exists():
        print(f"✗ Error: File not found: {descriptions_file}")
        sys.exit(1)

    # Support different file formats
    if descriptions_path.suffix == '.json':
        with open(descriptions_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                descriptions = data
            elif isinstance(data, dict) and 'descriptions' in data:
                descriptions = data['descriptions']
            else:
                print("✗ Error: JSON must be a list or have 'descriptions' key")
                sys.exit(1)
    elif descriptions_path.suffix in ['.txt', '.csv']:
        with open(descriptions_path, 'r') as f:
            descriptions = [line.strip() for line in f if line.strip()]
    else:
        print(f"✗ Error: Unsupported file format: {descriptions_path.suffix}")
        print("Supported formats: .txt, .csv, .json")
        sys.exit(1)

    if not descriptions:
        print("✗ Error: No descriptions found in file")
        sys.exit(1)

    print(f"\n Image Retrieval Batch Processing")
    print("=" * 50)
    print(f" Descriptions: {len(descriptions)}")
    print(f" Source: {source}")
    print(f" Output: {output_dir}")
    print(f" Preset: {preset}")
    print("=" * 50)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set up cache
    if cache_dir is None:
        cache_dir = os.path.expanduser('~/.nimitz_cache')

    # Initialize CLIP if needed
    model, device = None, None
    characteristics = None

    if use_clip or analyze:
        try:
            characteristics = get_preset_characteristics(preset)
            model, device = initialize_clip_model()
            if use_clip:
                print(" Using CLIP for image selection")
        except Exception as e:
            print(f" Warning: CLIP not available ({e})")
            use_clip = False

    # Retrieve images
    print(f"\n Retrieving images...")
    results = batch_retrieve_images(
        descriptions=descriptions,
        output_dir=output_dir,
        source_name=source,
        cache_dir=cache_dir,
        characteristics=characteristics if use_clip else None,
        model=model,
        device=device,
        num_candidates=5 if use_clip else 1
    )

    # Handle failed retrievals with placeholders
    successful = []
    failed = []

    for description, image_path, metadata in results:
        if image_path and os.path.exists(image_path):
            successful.append((description, image_path, metadata))
        else:
            failed.append((description, metadata))

            if use_placeholder:
                # Create placeholder image
                placeholder_path = os.path.join(
                    output_dir,
                    f"placeholder_{len(failed)}.jpg"
                )
                if create_placeholder_image(placeholder_path, description):
                    print(f" Created placeholder for: {description}")
                    successful.append((description, placeholder_path, {
                        'source': 'placeholder',
                        'description': description,
                        'error': metadata.get('error', 'Unknown error')
                    }))

    # Save metadata
    metadata_path = os.path.join(output_dir, 'retrieval_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump([
            {
                'description': desc,
                'image_path': path,
                'metadata': meta
            }
            for desc, path, meta in successful
        ], f, indent=2)

    print(f"\n Retrieval complete!")
    print(f"   Successful: {len(successful)}")
    print(f"   Failed: {len(failed)}")
    print(f"   Metadata: {metadata_path}")

    # Analyze images if requested
    if analyze and successful:
        print(f"\n Analyzing retrieved images...")

        image_paths = [path for _, path, _ in successful]

        if not characteristics:
            characteristics = get_preset_characteristics(preset)
        if not model:
            model, device = initialize_clip_model()

        # Extract features
        image_features = extract_image_features(image_paths, model, device)

        # Generate text features for all prompts
        all_prompts = []
        prompt_char_map = []
        for char_name, char_data in characteristics.items():
            for prompt in char_data['prompts']:
                all_prompts.append(prompt)
                prompt_char_map.append(char_name)

        text_features = extract_text_features(all_prompts, model, device)
        similarity_matrix = compute_similarity_matrices(
            image_features,
            text_features,
            characteristics
        )

        # Generate cards
        cards = generate_image_cards_data(
            image_paths,
            similarity_matrix,
            characteristics
        )

        # Add retrieval metadata to cards
        for i, card in enumerate(cards):
            if i < len(successful):
                desc, _, meta = successful[i]
                card['retrieval_metadata'] = {
                    'description': desc,
                    'source': meta.get('source', 'unknown'),
                    'author': meta.get('author'),
                    'license': meta.get('license'),
                    'attribution': meta.get('attribution'),
                    'source_url': meta.get('source_url')
                }

        # Enhance with gaming stats
        cards = enhance_cards_with_gaming_stats(cards)

        # Export results
        cards_json_path = os.path.join(output_dir, 'cards.json')
        with open(cards_json_path, 'w') as f:
            json.dump(cards, f, indent=2)

        cards_csv_path = os.path.join(output_dir, 'cards.csv')
        export_cards_to_csv(cards, cards_csv_path)

        print(f"\n Cards generated!")
        print(f"   JSON: {cards_json_path}")
        print(f"   CSV: {cards_csv_path}")

        # Create visual cards
        visual_output = os.path.join(output_dir, 'visual_cards')
        create_visual_image_cards(cards, visual_output)
        print(f"   Visual cards: {visual_output}/")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='nimitz',
        description='NIMITZ - Transform images into collectible cards with quantified statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  nimitz analyze ./photos --preset photography
  nimitz describe photo.jpg --preset art
  nimitz analyze ./products --preset products --output ./results
  nimitz presets

LLM Mode (no CLIP required):
  nimitz llm status                     # Check LLM provider availability
  nimitz llm describe photo.jpg         # Describe image with LLM
  nimitz llm analyze ./photos           # Analyze directory with LLM
  nimitz llm vocab ./photos -o vocab.json  # Generate vocabulary with LLM
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # -------------------------------------------------------------------------
    # analyze command
    # -------------------------------------------------------------------------
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze all images in a directory'
    )
    analyze_parser.add_argument(
        'directory',
        help='Directory containing images to analyze'
    )
    analyze_parser.add_argument(
        '-p', '--preset',
        choices=list_available_presets(),
        default='photography',
        help='Vocabulary preset (default: photography)'
    )
    analyze_parser.add_argument(
        '-o', '--output',
        help='Output directory (default: <directory>/nimitz_output)'
    )
    analyze_parser.add_argument(
        '-n', '--clusters',
        type=int,
        default=5,
        help='Number of clusters (default: 5)'
    )
    analyze_parser.add_argument(
        '--no-visual',
        action='store_true',
        help='Skip visual card generation (faster)'
    )
    analyze_parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Minimal output'
    )

    # -------------------------------------------------------------------------
    # describe command
    # -------------------------------------------------------------------------
    describe_parser = subparsers.add_parser(
        'describe',
        help='Analyze a single image'
    )
    describe_parser.add_argument(
        'image',
        help='Image file to analyze'
    )
    describe_parser.add_argument(
        '-p', '--preset',
        choices=list_available_presets(),
        default='photography',
        help='Vocabulary preset (default: photography)'
    )
    describe_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed analysis'
    )

    # -------------------------------------------------------------------------
    # presets command
    # -------------------------------------------------------------------------
    subparsers.add_parser(
        'presets',
        help='List available vocabulary presets'
    )

    # -------------------------------------------------------------------------
    # wizard command
    # -------------------------------------------------------------------------
    wizard_parser = subparsers.add_parser(
        'wizard',
        help='Interactive wizard to create custom vocabularies'
    )
    wizard_parser.add_argument(
        '-d', '--directory',
        help='Directory with sample images for suggestions'
    )
    wizard_parser.add_argument(
        '-o', '--output',
        help='Output file for saving the vocabulary (JSON)'
    )
    wizard_parser.add_argument(
        '--analyze',
        action='store_true',
        help='After creating vocabulary, analyze images immediately'
    )

    # -------------------------------------------------------------------------
    # validate command
    # -------------------------------------------------------------------------
    validate_parser = subparsers.add_parser(
        'validate',
        help='Validate a vocabulary file'
    )
    validate_parser.add_argument(
        'vocabulary_file',
        help='JSON file with vocabulary to validate'
    )

    # -------------------------------------------------------------------------
    # llm command (with subcommands)
    # -------------------------------------------------------------------------
    llm_parser = subparsers.add_parser(
        'llm',
        help='LLM-based analysis (no CLIP required)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands:
  status    Check LLM provider availability
  describe  Analyze a single image with LLM
  analyze   Analyze a directory of images with LLM
  vocab     Generate vocabulary from images using LLM

Examples:
  nimitz llm status
  nimitz llm describe photo.jpg
  nimitz llm analyze ./photos --provider anthropic
  nimitz llm vocab ./photos -o my_vocabulary.json
        """
    )
    llm_subparsers = llm_parser.add_subparsers(dest='llm_command', help='LLM subcommands')

    # llm status
    llm_subparsers.add_parser(
        'status',
        help='Check LLM provider availability'
    )

    # llm describe
    llm_describe_parser = llm_subparsers.add_parser(
        'describe',
        help='Analyze a single image with LLM'
    )
    llm_describe_parser.add_argument(
        'image',
        help='Image file to analyze'
    )
    llm_describe_parser.add_argument(
        '--provider',
        choices=['openai', 'anthropic', 'gemini', 'auto'],
        default='auto',
        help='LLM provider (default: auto)'
    )
    llm_describe_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed analysis'
    )
    llm_describe_parser.add_argument(
        '--lang',
        choices=['en', 'it'],
        default='en',
        help='Response language (default: en)'
    )

    # llm analyze
    llm_analyze_parser = llm_subparsers.add_parser(
        'analyze',
        help='Analyze a directory of images with LLM'
    )
    llm_analyze_parser.add_argument(
        'directory',
        help='Directory containing images to analyze'
    )
    llm_analyze_parser.add_argument(
        '-p', '--preset',
        choices=list_available_presets(),
        default='photography',
        help='Vocabulary preset (default: photography)'
    )
    llm_analyze_parser.add_argument(
        '-o', '--output',
        help='Output directory'
    )
    llm_analyze_parser.add_argument(
        '--provider',
        choices=['openai', 'anthropic', 'gemini', 'auto'],
        default='auto',
        help='LLM provider (default: auto)'
    )
    llm_analyze_parser.add_argument(
        '--no-visual',
        action='store_true',
        help='Skip visual card generation'
    )
    llm_analyze_parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Minimal output'
    )
    llm_analyze_parser.add_argument(
        '--lang',
        choices=['en', 'it'],
        default='en',
        help='Response language (default: en)'
    )

    # llm vocab
    llm_vocab_parser = llm_subparsers.add_parser(
        'vocab',
        help='Generate vocabulary from images using LLM'
    )
    llm_vocab_parser.add_argument(
        'directory',
        help='Directory with sample images'
    )
    llm_vocab_parser.add_argument(
        '-o', '--output',
        help='Output file for vocabulary (JSON)'
    )
    llm_vocab_parser.add_argument(
        '--provider',
        choices=['openai', 'anthropic', 'gemini', 'auto'],
        default='auto',
        help='LLM provider (default: auto)'
    )
    llm_vocab_parser.add_argument(
        '-n', '--samples',
        type=int,
        default=5,
        help='Number of images to sample (default: 5)'
    )
    llm_vocab_parser.add_argument(
        '--lang',
        choices=['en', 'it'],
        default='en',
        help='Response language (default: en)'
    )

    # -------------------------------------------------------------------------
    # compare command (Gaming)
    # -------------------------------------------------------------------------
    compare_parser = subparsers.add_parser(
        'compare',
        help='Compare two cards to see who wins'
    )
    compare_parser.add_argument(
        'cards_file',
        help='JSON file with card data (from nimitz analyze)'
    )
    compare_parser.add_argument(
        'card1',
        help='First card name (partial match supported)'
    )
    compare_parser.add_argument(
        'card2',
        help='Second card name (partial match supported)'
    )
    compare_parser.add_argument(
        '-c', '--category',
        help='Specific characteristic to compare (default: overall power)'
    )

    # -------------------------------------------------------------------------
    # battle command (Gaming)
    # -------------------------------------------------------------------------
    battle_parser = subparsers.add_parser(
        'battle',
        help='Full battle between two cards across multiple rounds'
    )
    battle_parser.add_argument(
        'cards_file',
        help='JSON file with card data'
    )
    battle_parser.add_argument(
        'card1',
        help='First card name'
    )
    battle_parser.add_argument(
        'card2',
        help='Second card name'
    )
    battle_parser.add_argument(
        '-r', '--rounds',
        type=int,
        default=5,
        help='Number of battle rounds (default: 5)'
    )

    # -------------------------------------------------------------------------
    # rarity command (Gaming)
    # -------------------------------------------------------------------------
    rarity_parser = subparsers.add_parser(
        'rarity',
        help='Show card rarity rankings'
    )
    rarity_parser.add_argument(
        'cards_file',
        help='JSON file with card data'
    )
    rarity_parser.add_argument(
        '-n', '--top',
        type=int,
        default=10,
        help='Number of cards to show (default: 10)'
    )

    # -------------------------------------------------------------------------
    # stats command (Gaming)
    # -------------------------------------------------------------------------
    stats_parser = subparsers.add_parser(
        'stats',
        help='Show collection statistics'
    )
    stats_parser.add_argument(
        'cards_file',
        help='JSON file with card data'
    )

    # -------------------------------------------------------------------------
    # deck command (Gaming - with subcommands)
    # -------------------------------------------------------------------------
    deck_parser = subparsers.add_parser(
        'deck',
        help='Deck management (create, show, add, remove cards)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands:
  create    Create a new deck from cards
  show      Show deck information
  add       Add cards to a deck
  remove    Remove cards from a deck

Examples:
  nimitz deck create cards.json --name "My Deck" -o my_deck.json
  nimitz deck create cards.json --top 10 --name "Top 10" -o top_deck.json
  nimitz deck show my_deck.json
  nimitz deck show my_deck.json --list
  nimitz deck add my_deck.json cards.json photo1.jpg photo2.jpg
  nimitz deck remove my_deck.json photo1.jpg
        """
    )
    deck_subparsers = deck_parser.add_subparsers(dest='deck_command', help='Deck subcommands')

    # deck create
    deck_create_parser = deck_subparsers.add_parser(
        'create',
        help='Create a new deck'
    )
    deck_create_parser.add_argument(
        'cards_file',
        help='JSON file with card data to choose from'
    )
    deck_create_parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output file for the deck'
    )
    deck_create_parser.add_argument(
        '-n', '--name',
        default='My Deck',
        help='Deck name (default: My Deck)'
    )
    deck_create_parser.add_argument(
        '--top',
        type=int,
        help='Select top N cards by power level'
    )
    deck_create_parser.add_argument(
        '--cards',
        nargs='+',
        help='Specific card names to include'
    )

    # deck show
    deck_show_parser = deck_subparsers.add_parser(
        'show',
        help='Show deck information'
    )
    deck_show_parser.add_argument(
        'deck_file',
        help='Deck file to show'
    )
    deck_show_parser.add_argument(
        '-l', '--list',
        action='store_true',
        help='List all cards in deck'
    )

    # deck add
    deck_add_parser = deck_subparsers.add_parser(
        'add',
        help='Add cards to a deck'
    )
    deck_add_parser.add_argument(
        'deck_file',
        help='Deck file to modify'
    )
    deck_add_parser.add_argument(
        'cards_file',
        help='JSON file with card data'
    )
    deck_add_parser.add_argument(
        'card_names',
        nargs='+',
        help='Card names to add'
    )

    # deck remove
    deck_remove_parser = deck_subparsers.add_parser(
        'remove',
        help='Remove cards from a deck'
    )
    deck_remove_parser.add_argument(
        'deck_file',
        help='Deck file to modify'
    )
    deck_remove_parser.add_argument(
        'card_names',
        nargs='+',
        help='Card names to remove'
    )

    # -------------------------------------------------------------------------
    # export-pdf command (Gaming)
    # -------------------------------------------------------------------------
    export_pdf_parser = subparsers.add_parser(
        'export-pdf',
        help='Export cards as printable PDF'
    )
    export_pdf_parser.add_argument(
        'cards_file',
        help='JSON file with card data (or deck file)'
    )
    export_pdf_parser.add_argument(
        '-o', '--output',
        default='cards.pdf',
        help='Output PDF file (default: cards.pdf)'
    )
    export_pdf_parser.add_argument(
        '--size',
        choices=['A4', 'Letter'],
        default='A4',
        help='Page size (default: A4)'
    )

    # -------------------------------------------------------------------------
    # export-png command (Gaming)
    # -------------------------------------------------------------------------
    export_png_parser = subparsers.add_parser(
        'export-png',
        help='Export cards as individual PNG files'
    )
    export_png_parser.add_argument(
        'cards_file',
        help='JSON file with card data'
    )
    export_png_parser.add_argument(
        '-o', '--output',
        default='./card_images',
        help='Output directory (default: ./card_images)'
    )

    # -------------------------------------------------------------------------
    # retrieve command (Image Retrieval)
    # -------------------------------------------------------------------------
    retrieve_parser = subparsers.add_parser(
        'retrieve',
        help='Retrieve images from the web and generate cards',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Image Retrieval: Create cards from text descriptions by fetching images from the web.

Subcommands:
  status   Check API configuration status
  single   Retrieve a single image
  batch    Retrieve multiple images from file

Examples:
  nimitz retrieve status
  nimitz retrieve single "Golden Gate Bridge at sunset"
  nimitz retrieve batch players.txt --preset art
        """
    )
    retrieve_subparsers = retrieve_parser.add_subparsers(dest='retrieve_command', help='Retrieve subcommands')

    # retrieve status
    retrieve_subparsers.add_parser(
        'status',
        help='Check image retrieval API configuration'
    )

    # retrieve single
    retrieve_single_parser = retrieve_subparsers.add_parser(
        'single',
        help='Retrieve a single image by description'
    )
    retrieve_single_parser.add_argument(
        'description',
        help='Text description of the image to retrieve'
    )
    retrieve_single_parser.add_argument(
        '-o', '--output',
        default='./nimitz_output',
        help='Output directory (default: ./nimitz_output)'
    )
    retrieve_single_parser.add_argument(
        '-p', '--preset',
        default='photography',
        help='Vocabulary preset (default: photography)'
    )
    retrieve_single_parser.add_argument(
        '-s', '--source',
        choices=['unsplash', 'pexels'],
        default='unsplash',
        help='Image source (default: unsplash)'
    )
    retrieve_single_parser.add_argument(
        '--cache',
        help='Cache directory (default: ~/.nimitz_cache)'
    )
    retrieve_single_parser.add_argument(
        '--no-clip',
        action='store_true',
        help='Disable CLIP-based image selection'
    )
    retrieve_single_parser.add_argument(
        '--no-analyze',
        action='store_true',
        help='Skip image analysis after retrieval'
    )

    # retrieve batch
    retrieve_batch_parser = retrieve_subparsers.add_parser(
        'batch',
        help='Retrieve multiple images from a file'
    )
    retrieve_batch_parser.add_argument(
        'descriptions_file',
        help='File with descriptions (.txt, .csv, or .json)'
    )
    retrieve_batch_parser.add_argument(
        '-o', '--output',
        default='./nimitz_output',
        help='Output directory (default: ./nimitz_output)'
    )
    retrieve_batch_parser.add_argument(
        '-p', '--preset',
        default='photography',
        help='Vocabulary preset (default: photography)'
    )
    retrieve_batch_parser.add_argument(
        '-s', '--source',
        choices=['unsplash', 'pexels'],
        default='unsplash',
        help='Image source (default: unsplash)'
    )
    retrieve_batch_parser.add_argument(
        '--cache',
        help='Cache directory (default: ~/.nimitz_cache)'
    )
    retrieve_batch_parser.add_argument(
        '--no-clip',
        action='store_true',
        help='Disable CLIP-based image selection'
    )
    retrieve_batch_parser.add_argument(
        '--no-analyze',
        action='store_true',
        help='Skip card generation (only retrieve images)'
    )
    retrieve_batch_parser.add_argument(
        '--no-placeholder',
        action='store_true',
        help='Skip creating placeholder images for failed retrievals'
    )

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Execute command
    if args.command == 'analyze':
        analyze_directory(
            directory=args.directory,
            preset=args.preset,
            output_dir=args.output,
            n_clusters=args.clusters,
            no_visual=args.no_visual,
            quiet=args.quiet
        )

    elif args.command == 'describe':
        describe_single_image(
            image_path=args.image,
            preset=args.preset,
            verbose=args.verbose
        )

    elif args.command == 'presets':
        list_presets()

    elif args.command == 'wizard':
        run_vocabulary_wizard(
            image_directory=args.directory,
            output_file=args.output,
            analyze_after=args.analyze
        )

    elif args.command == 'validate':
        validate_vocabulary_file(args.vocabulary_file)

    elif args.command == 'llm':
        if args.llm_command is None:
            llm_parser.print_help()
            sys.exit(0)

        elif args.llm_command == 'status':
            print_llm_status()

        elif args.llm_command == 'describe':
            describe_single_image_llm(
                image_path=args.image,
                provider=args.provider,
                verbose=args.verbose,
                language=args.lang
            )

        elif args.llm_command == 'analyze':
            analyze_directory_llm(
                directory=args.directory,
                preset=args.preset,
                output_dir=args.output,
                provider=args.provider,
                no_visual=args.no_visual,
                quiet=args.quiet,
                language=args.lang
            )

        elif args.llm_command == 'vocab':
            generate_vocabulary_llm(
                directory=args.directory,
                output_file=args.output,
                provider=args.provider,
                num_samples=args.samples,
                language=args.lang
            )

    # -------------------------------------------------------------------------
    # Gaming commands
    # -------------------------------------------------------------------------
    elif args.command == 'compare':
        compare_cards_command(
            cards_file=args.cards_file,
            card1_name=args.card1,
            card2_name=args.card2,
            category=args.category
        )

    elif args.command == 'battle':
        battle_cards_command(
            cards_file=args.cards_file,
            card1_name=args.card1,
            card2_name=args.card2,
            rounds=args.rounds
        )

    elif args.command == 'rarity':
        show_rarity_command(
            cards_file=args.cards_file,
            top_n=args.top
        )

    elif args.command == 'stats':
        show_stats_command(args.cards_file)

    elif args.command == 'deck':
        if args.deck_command is None:
            deck_parser.print_help()
            sys.exit(0)

        elif args.deck_command == 'create':
            deck_create_command(
                cards_file=args.cards_file,
                deck_name=args.name,
                output_file=args.output,
                card_names=args.cards,
                top_n=args.top
            )

        elif args.deck_command == 'show':
            deck_show_command(
                deck_file=args.deck_file,
                list_cards=args.list
            )

        elif args.deck_command == 'add':
            deck_add_command(
                deck_file=args.deck_file,
                cards_file=args.cards_file,
                card_names=args.card_names
            )

        elif args.deck_command == 'remove':
            deck_remove_command(
                deck_file=args.deck_file,
                card_names=args.card_names
            )

    elif args.command == 'export-pdf':
        export_pdf_command(
            cards_file=args.cards_file,
            output_file=args.output,
            page_size=args.size
        )

    elif args.command == 'export-png':
        export_png_command(
            cards_file=args.cards_file,
            output_dir=args.output
        )

    # -------------------------------------------------------------------------
    # Image Retrieval commands
    # -------------------------------------------------------------------------
    elif args.command == 'retrieve':
        if args.retrieve_command is None:
            retrieve_parser.print_help()
            sys.exit(0)

        elif args.retrieve_command == 'status':
            retrieve_status_command()

        elif args.retrieve_command == 'single':
            retrieve_single_command(
                description=args.description,
                output_dir=args.output,
                preset=args.preset,
                source=args.source,
                cache_dir=args.cache,
                use_clip=not args.no_clip,
                analyze=not args.no_analyze
            )

        elif args.retrieve_command == 'batch':
            retrieve_batch_command(
                descriptions_file=args.descriptions_file,
                output_dir=args.output,
                preset=args.preset,
                source=args.source,
                cache_dir=args.cache,
                use_clip=not args.no_clip,
                analyze=not args.no_analyze,
                use_placeholder=not args.no_placeholder
            )


if __name__ == '__main__':
    main()
