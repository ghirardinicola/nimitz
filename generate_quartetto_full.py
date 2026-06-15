#!/usr/bin/env python3
"""
NIMITZ Quartetto Cards - Complete Generation Pipeline
Downloads Wikipedia images and generates all card formats
"""

import sys
import json
from pathlib import Path
from src.wikimedia_retrieval import WikimediaRetriever
from src.card_renderer_quartetto import (
    generate_all_cards,
    generate_print_pages,
    export_to_pdf,
)


def main():
    print("\n" + "=" * 70)
    print("  NIMITZ QUARTETTO CARDS - COMPLETE GENERATION PIPELINE")
    print("=" * 70)

    # Configuration
    analysis_file = "informatici_cards_analysis.json"
    enriched_file = "informatici_cards_analysis_enriched.json"
    wiki_dir = "./cards/quartetto/wiki_images"
    output_dir = "cards/quartetto"

    # Step 1: Check if enriched JSON exists
    if not Path(enriched_file).exists():
        print(f"\n❌ Enriched JSON not found: {enriched_file}")
        print("   Run: python3 enrich_cards_json.py first")
        sys.exit(1)

    print(f"\n✓ Using enriched data: {enriched_file}")

    # Step 2: Download Wikipedia images (if needed)
    wiki_path = Path(wiki_dir)
    if not wiki_path.exists() or len(list(wiki_path.glob("*.{png,jpg,jpeg,JPG}"))) < 32:
        print(f"\n📥 STEP 1: Downloading Wikipedia images...")
        print("=" * 70)

        with open(analysis_file) as f:
            cards = json.load(f)
        names = [card["name"] for card in cards]

        retriever = WikimediaRetriever(cache_dir=f"{wiki_dir}/metadata")
        results = retriever.batch_retrieve(names, wiki_dir, delay=1.0)

        successful = sum(1 for r in results if r["success"])
        print(f"\n✓ Downloaded/created {successful}/{len(names)} images")
    else:
        print(f"\n✓ Wikipedia images already downloaded")

    # Step 3: Re-enrich JSON with Wikipedia images
    print(f"\n🔄 STEP 2: Refreshing enriched JSON with Wikipedia images...")
    print("=" * 70)

    import subprocess

    result = subprocess.run(
        ["python3", "enrich_cards_json.py"], capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"❌ Error enriching JSON:")
        print(result.stderr)
        sys.exit(1)

    print("✓ Enriched JSON updated")

    # Step 4: Generate display cards
    print(f"\n🎨 STEP 3: Generating display cards (600×850)...")
    print("=" * 70)

    generate_all_cards(
        json_file=enriched_file,
        output_dir=output_dir,
        card_size="display",
        generate_backs=True,
    )

    # Step 5: Generate poker cards
    print(f"\n🎨 STEP 4: Generating poker cards (744×1039)...")
    print("=" * 70)

    generate_all_cards(
        json_file=enriched_file,
        output_dir=output_dir,
        card_size="poker",
        generate_backs=True,
    )

    # Step 6: Generate print pages
    print(f"\n🖨️  STEP 5: Generating 3×3 print pages...")
    print("=" * 70)

    generate_print_pages(
        json_file=enriched_file,
        output_dir=output_dir,
        card_size="poker",
        cards_per_page=9,
        generate_backs=True,
    )

    # Step 7: Generate PDF
    print(f"\n📄 STEP 6: Generating print-ready PDF...")
    print("=" * 70)

    pdf_path = f"{output_dir}/poker/NIMITZ_Cards_Printable.pdf"
    export_to_pdf(
        json_file=enriched_file,
        output_path=pdf_path,
        card_size="poker",
        include_backs=True,
    )

    # Summary
    print("\n" + "=" * 70)
    print("  ✅ COMPLETE! All cards generated successfully")
    print("=" * 70)

    print(f"\n📁 Output directories:")
    print(f"   Display cards: {output_dir}/display/fronts/")
    print(f"   Poker cards:   {output_dir}/poker/fronts/")
    print(f"   Print pages:   {output_dir}/poker/print_pages/")
    print(f"   PDF:           {pdf_path}")

    print(f"\n📊 Statistics:")
    with open(enriched_file) as f:
        cards = json.load(f)

    wikipedia = sum(1 for c in cards if c.get("image_source") == "wikipedia")
    placeholder = sum(1 for c in cards if c.get("image_source") == "placeholder")
    print(f"   Total cards:        32")
    print(f"   Wikipedia images:   {wikipedia}")
    print(f"   Placeholder images: {placeholder}")
    print(f"   Display cards:      32 fronts + 1 back")
    print(f"   Poker cards:        32 fronts + 1 back")
    print(f"   Print pages:        4 fronts + 4 backs (3×3 grid)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
