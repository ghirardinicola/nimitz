#!/usr/bin/env python3
"""
Quick test script for LLM filtering functionality.
Tests filter_names_with_llm() with mock data.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from web_discovery import filter_names_with_llm

# Mock data similar to real "baseball italiani" results
test_entities = [
    "Mike Piazza",
    "Baseball Almanac",
    "Joe DiMaggio",
    "Complete List",
    "Current Players",
    "Anthony Rizzo",
    "Carolina League",
    "First Base Jim Adduci",
    "Nick Punto",
    "Italian Baseball League",
    "Frank Viola",
    "Minor League Baseball",
]

print("🧪 Testing LLM Filter")
print("=" * 60)
print(f"\n📋 Input: {len(test_entities)} entities")
for i, entity in enumerate(test_entities, 1):
    print(f"  {i:2d}. {entity}")

print("\n🤖 Calling filter_names_with_llm()...")
print("   (This will use LLM API - make sure you have API key set)")

try:
    person_names, filtered_out = filter_names_with_llm(
        entities=test_entities, context="baseball players", provider="auto"
    )

    print("\n✅ SUCCESS!")
    print("=" * 60)
    print(f"\n👤 Person names ({len(person_names)}):")
    for i, name in enumerate(person_names, 1):
        print(f"  {i:2d}. {name}")

    print(f"\n🚫 Filtered out ({len(filtered_out)}):")
    for i, item in enumerate(filtered_out, 1):
        print(f"  {i:2d}. {item}")

    print("\n" + "=" * 60)
    print("✅ Test completed successfully!")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nMake sure you have one of these environment variables set:")
    print("  - ANTHROPIC_API_KEY")
    print("  - GEMINI_API_KEY")
    print("  - OPENAI_API_KEY")
    sys.exit(1)
