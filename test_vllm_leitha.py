#!/usr/bin/env python3
"""
Test rapido per verificare la connessione al server vLLM Leitha
e il funzionamento del filtering LLM.
"""

import os
import sys

# Configura le variabili d'ambiente per il test
os.environ["VLLM_BASE_URL"] = "https://agent-codeai.leitha.servizi.gr-u.it/v1"
os.environ["VLLM_API_KEY"] = "anything"
os.environ["VLLM_MODEL"] = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

print("🧪 Test NIMITZ con vLLM Leitha")
print("=" * 60)

# Test 1: Verifica configurazione
print("\n📋 Step 1: Verifica configurazione")
print(f"  ✓ VLLM_BASE_URL: {os.environ['VLLM_BASE_URL']}")
print(f"  ✓ VLLM_API_KEY: {os.environ['VLLM_API_KEY']}")
print(f"  ✓ VLLM_MODEL: {os.environ['VLLM_MODEL']}")

# Test 2: Importa moduli
print("\n📋 Step 2: Import moduli NIMITZ")
try:
    from llm_analyzer import get_llm_config

    print("  ✓ llm_analyzer importato")
except ImportError as e:
    print(f"  ✗ Errore import llm_analyzer: {e}")
    sys.exit(1)

try:
    from web_discovery import filter_names_with_llm

    print("  ✓ web_discovery importato")
except ImportError as e:
    print(f"  ✗ Errore import web_discovery: {e}")
    sys.exit(1)

# Test 3: Ottieni configurazione LLM
print("\n📋 Step 3: Configurazione LLM")
try:
    config = get_llm_config("vllm")
    print(f"  ✓ Provider: {config.provider}")
    print(f"  ✓ Model: {config.model}")
    print(f"  ✓ Base URL: {config.base_url}")
    print(f"  ✓ API Key: {'*' * len(config.api_key) if config.api_key else 'None'}")
except Exception as e:
    print(f"  ✗ Errore: {e}")
    sys.exit(1)

# Test 4: Test filtering con dati di esempio
print("\n📋 Step 4: Test LLM filtering")
test_entities = [
    "Mike Piazza",
    "Baseball Almanac",
    "Joe DiMaggio",
    "New York Yankees",
    "Anthony Rizzo",
    "Complete List",
]

print(f"\n  Input: {len(test_entities)} entities")
for i, entity in enumerate(test_entities, 1):
    print(f"    {i}. {entity}")

print("\n  🤖 Chiamata al server vLLM Leitha...")
print("     (questo potrebbe richiedere alcuni secondi)")

try:
    person_names, filtered_out = filter_names_with_llm(
        entities=test_entities, context="baseball players", provider="vllm"
    )

    print("\n  ✅ SUCCESS!")
    print("  " + "=" * 56)

    print(f"\n  👤 Person names ({len(person_names)}):")
    for i, name in enumerate(person_names, 1):
        print(f"    {i}. {name}")

    print(f"\n  🚫 Filtered out ({len(filtered_out)}):")
    for i, item in enumerate(filtered_out, 1):
        print(f"    {i}. {item}")

    print("\n" + "=" * 60)
    print("✅ Test completato con successo!")
    print("\nIl server vLLM Leitha è configurato correttamente.")
    print("Puoi ora usare NIMITZ con --llm-provider vllm")

except Exception as e:
    print(f"\n  ❌ ERRORE: {e}")
    print("\n  Possibili cause:")
    print("    - Server vLLM non raggiungibile")
    print("    - Credenziali non corrette")
    print("    - Problema di rete/VPN")
    print("\n  Debug:")
    import traceback

    traceback.print_exc()
    sys.exit(1)
