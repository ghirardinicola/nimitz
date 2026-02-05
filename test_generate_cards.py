#!/usr/bin/env python3
"""
Test script per generare card grafiche da JSON esistente
Utile per rigenerare le card senza rifare l'analisi
"""

import json
import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from image_card import create_visual_image_cards


def load_cards_from_json(json_file: str) -> list:
    """Carica i dati delle card dal JSON"""
    with open(json_file) as f:
        return json.load(f)


def convert_to_visual_format(cards_data: list) -> list:
    """
    Converte dal formato enriched al formato richiesto da create_visual_image_cards
    """
    cards_visual = []

    for card in cards_data:
        # Create dominant features from scores
        dominant_features = []
        for char_name, score in sorted(
            card["scores"].items(), key=lambda x: x[1], reverse=True
        ):
            dominant_features.append(
                {
                    "characteristic": char_name,
                    "prompt": char_name.replace("_", " ").title(),
                    "score": score / 100.0,  # Convert to 0-1 scale
                    "confidence": "high" if score > 70 else "medium",
                }
            )

        # Calculate feature summary
        scores_list = list(card["scores"].values())
        scores_normalized = [s / 100.0 for s in scores_list]

        card_visual = {
            "image_name": card["name"],
            "image_path": card["image"],
            "image_index": 0,
            "characteristics": {},
            "feature_summary": {
                "total_features": len(scores_list),
                "overall_max": max(scores_normalized),
                "overall_mean": sum(scores_normalized) / len(scores_normalized),
                "overall_std": 0.0,
                "high_confidence_features": sum(1 for s in scores_list if s > 70),
                "medium_confidence_features": sum(
                    1 for s in scores_list if 40 <= s <= 70
                ),
                "low_confidence_features": sum(1 for s in scores_list if s < 40),
            },
            "dominant_features": dominant_features[:5],
        }

        cards_visual.append(card_visual)

    return cards_visual


def main():
    print("\n" + "=" * 70)
    print("  TEST GENERAZIONE CARD GRAFICHE")
    print("=" * 70)

    # Check if JSON file exists
    json_file = "informatici_cards_analysis.json"
    if not Path(json_file).exists():
        print(f"\n❌ File {json_file} non trovato!")
        print("   Esegui prima il workflow: ./run_workflow.sh")
        sys.exit(1)

    # Load cards
    print(f"\n📖 Caricamento dati da {json_file}...")
    cards_data = load_cards_from_json(json_file)
    print(f"✓ Caricate {len(cards_data)} carte")

    # Convert to visual format
    print("\n🔄 Conversione formato dati...")
    cards_visual = convert_to_visual_format(cards_data)
    print(f"✓ Convertite {len(cards_visual)} carte")

    # Generate visual cards
    print("\n🎨 Generazione card grafiche...")
    output_dir = "./informatici_cards_visual"

    try:
        create_visual_image_cards(
            cards_data=cards_visual,
            output_dir=output_dir,
            cards_per_page=6,
            show_thumbnails=True,
        )

        print(f"\n✅ Card grafiche generate con successo!")
        print(f"   Directory output: {output_dir}")
        print(f"   - Pagine: {output_dir}/image_cards_page_*.png")
        print(f"   - Individuali: {output_dir}/individual/")

    except Exception as e:
        print(f"\n❌ Errore durante la generazione: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
