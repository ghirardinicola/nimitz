#!/usr/bin/env python3
"""
NIMITZ - Command Line Interface
Usage-friendly CLI for image analysis and card generation
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any

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


if __name__ == '__main__':
    main()
