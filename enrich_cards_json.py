#!/usr/bin/env python3
"""
NIMITZ - JSON Enrichment with Descriptions
Adds human-readable descriptions to card scores based on vocabulary
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import os
import glob


def load_json(filepath: str) -> Dict:
    """Load JSON file"""
    with open(filepath) as f:
        return json.load(f)


def format_display_name(key: str) -> str:
    """Format characteristic key to display name"""
    # Use display_names from vocab if available, otherwise format key
    return key.replace("_", " ").title()


def map_italian_to_english_keys(scores: Dict, vocab_en: Dict) -> Dict:
    """Map Italian characteristic names to English"""
    mapping = {
        "influenza_sul_mercato": "market_influence",
        "uso_di_tecnologie_avanzate": "advanced_technologies",
        "riconoscimento_professionale": "professional_recognition",
        "lunghezza_della_barba": "beard_length",
        "openess": "openness",
    }

    new_scores = {}
    for it_key, value in scores.items():
        en_key = mapping.get(it_key, it_key)
        new_scores[en_key] = value

    return new_scores


def get_wikipedia_image_info(
    scientist_name: str, wiki_dir: str = "./cards/quartetto/wiki_images"
) -> Dict:
    """
    Get Wikipedia image path and attribution for a scientist

    Args:
        scientist_name: Name of the scientist
        wiki_dir: Directory containing Wikipedia images

    Returns:
        Dict with image_path, image_source, attribution
    """
    # Search for image file (could be .png, .jpg, .jpeg, .JPG)
    name_safe = scientist_name.replace(" ", "_")
    patterns = [
        f"{wiki_dir}/{name_safe}.png",
        f"{wiki_dir}/{name_safe}.jpg",
        f"{wiki_dir}/{name_safe}.jpeg",
        f"{wiki_dir}/{name_safe}.JPG",
    ]

    image_path = None
    for pattern in patterns:
        if os.path.exists(pattern):
            image_path = pattern
            break

    if not image_path:
        # Fallback to original image if Wikipedia not found
        return {"image_path": None, "image_source": "pexels", "attribution": "Pexels"}

    # Check for metadata file
    metadata_path = f"{wiki_dir}/metadata/{name_safe}.json"
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Check if it's a placeholder or real Wikipedia image
        if metadata.get("image_url"):
            return {
                "image_path": image_path,
                "image_source": "wikipedia",
                "attribution": metadata.get("attribution", "Wikimedia Commons"),
                "page_url": metadata.get("page_url", ""),
            }
        else:
            # It's a placeholder
            return {
                "image_path": image_path,
                "image_source": "placeholder",
                "attribution": "Photo unavailable",
            }

    # No metadata, assume Wikipedia
    return {
        "image_path": image_path,
        "image_source": "wikipedia",
        "attribution": "Wikimedia Commons",
    }


def enrich_card_with_descriptions(
    card: Dict, vocab: Dict, wiki_dir: str = "./cards/quartetto/wiki_images"
) -> Dict:
    """
    Enrich card data with descriptions and rankings

    Args:
        card: Card dict with name, image, scores
        vocab: Vocabulary dict with characteristics
        wiki_dir: Directory containing Wikipedia images

    Returns:
        Enriched card dict
    """
    # Convert Italian keys to English if needed
    scores_dict = card.get("scores", {})
    if "influenza_sul_mercato" in scores_dict:
        scores_dict = map_italian_to_english_keys(scores_dict, vocab)

    # Get display names
    display_names = vocab.get("display_names", {})

    # Sort scores by value (descending) to get ranks
    sorted_scores = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)

    # Enrich each score
    enriched_scores = {}

    for rank, (char_key, score_value) in enumerate(sorted_scores, 1):
        # Map score (0-100) to description level (0-4)
        level = min(
            int(score_value / 20), 4
        )  # 0-19→0, 20-39→1, 40-59→2, 60-79→3, 80-100→4

        # Get description from vocabulary
        descriptions = vocab["characteristics"].get(char_key, [])
        if level < len(descriptions):
            description = descriptions[level]
        else:
            description = f"Score: {score_value}/100"

        # Get display name
        display_name = display_names.get(char_key, format_display_name(char_key))

        # Create enriched score object
        enriched_scores[char_key] = {
            "value": score_value,
            "level": level,
            "description": description,
            "rank": rank,
            "display_name": display_name,
        }

    # Identify top skill (highest score)
    top_char_key, top_char_value = sorted_scores[0]
    top_skill = {
        "key": top_char_key,
        "display_name": enriched_scores[top_char_key]["display_name"],
        "value": top_char_value,
        "description": enriched_scores[top_char_key]["description"],
    }

    # Get Wikipedia image info
    wiki_info = get_wikipedia_image_info(card["name"], wiki_dir)

    # Use Wikipedia image if available, otherwise keep original
    image_path = wiki_info["image_path"] if wiki_info["image_path"] else card["image"]

    # Create enriched card
    enriched_card = {
        "name": card["name"],
        "image": image_path,
        "image_source": wiki_info["image_source"],
        "attribution": wiki_info["attribution"],
        "scores": enriched_scores,
        "top_skill": top_skill,
    }

    # Add page_url if available
    if "page_url" in wiki_info:
        enriched_card["page_url"] = wiki_info["page_url"]

    return enriched_card


def enrich_all_cards(
    cards_data: List[Dict], vocab: Dict, wiki_dir: str = "./cards/quartetto/wiki_images"
) -> List[Dict]:
    """Enrich all cards in dataset"""
    enriched_cards = []

    for card in cards_data:
        enriched = enrich_card_with_descriptions(card, vocab, wiki_dir)
        enriched_cards.append(enriched)

    return enriched_cards


def main():
    print("\n" + "=" * 70)
    print("  NIMITZ - JSON Enrichment Test")
    print("=" * 70)

    # Load existing analysis
    analysis_file = "informatici_cards_analysis.json"
    if not Path(analysis_file).exists():
        print(f"\n❌ File not found: {analysis_file}")
        print("   Run the workflow first: ./run_workflow.sh")
        sys.exit(1)

    print(f"\n📖 Loading: {analysis_file}")
    cards_data = load_json(analysis_file)
    print(f"✓ Loaded {len(cards_data)} cards")

    # Load English vocabulary
    vocab_file = "vocabolario_informatici_game_en.json"
    if not Path(vocab_file).exists():
        print(f"\n❌ File not found: {vocab_file}")
        sys.exit(1)

    print(f"\n📖 Loading: {vocab_file}")
    vocab = load_json(vocab_file)
    print(f"✓ Loaded vocabulary with {len(vocab['characteristics'])} characteristics")

    # Check for Wikipedia images
    wiki_dir = "./cards/quartetto/wiki_images"
    wiki_exists = Path(wiki_dir).exists()
    if wiki_exists:
        wiki_count = len(list(Path(wiki_dir).glob("*.{png,jpg,jpeg,JPG}")))
        print(f"✓ Found Wikipedia images directory with {wiki_count} images")
    else:
        print(f"⚠️  Wikipedia images not found, using original images")

    # Enrich cards
    print(f"\n🔄 Enriching {len(cards_data)} cards...")
    enriched_cards = enrich_all_cards(cards_data, vocab, wiki_dir)
    print(f"✓ Enriched {len(enriched_cards)} cards")

    # Count image sources
    wikipedia_count = sum(
        1 for c in enriched_cards if c.get("image_source") == "wikipedia"
    )
    placeholder_count = sum(
        1 for c in enriched_cards if c.get("image_source") == "placeholder"
    )
    other_count = len(enriched_cards) - wikipedia_count - placeholder_count

    print(f"\n📸 Image sources:")
    print(f"   Wikipedia: {wikipedia_count}")
    print(f"   Placeholder: {placeholder_count}")
    print(f"   Other: {other_count}")

    # Show sample
    print("\n" + "=" * 70)
    print("  SAMPLE ENRICHED CARD")
    print("=" * 70)

    sample = enriched_cards[0]
    print(f"\n👤 Name: {sample['name']}")
    print(f"📸 Image: {sample['image']}")
    print(f"\n🏆 TOP SKILL: {sample['top_skill']['display_name']}")
    print(f"   Value: {sample['top_skill']['value']}/100")
    print(f'   Description: "{sample["top_skill"]["description"]}"')

    print(f"\n📊 ALL SCORES:")
    for char_key, char_data in sorted(
        sample["scores"].items(), key=lambda x: x[1]["rank"]
    ):
        rank_emoji = (
            "🥇"
            if char_data["rank"] == 1
            else "🥈"
            if char_data["rank"] == 2
            else "🥉"
            if char_data["rank"] == 3
            else "  "
        )
        print(
            f"   {rank_emoji} #{char_data['rank']} {char_data['display_name']}: {char_data['value']}/100"
        )
        print(f'      Level {char_data["level"]}: "{char_data["description"]}"')

    # Save enriched data
    output_file = "informatici_cards_analysis_enriched.json"
    print(f"\n💾 Saving enriched data to: {output_file}")

    with open(output_file, "w") as f:
        json.dump(enriched_cards, f, indent=2)

    print(f"✓ Saved {len(enriched_cards)} enriched cards")

    print("\n" + "=" * 70)
    print("✅ Enrichment complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
