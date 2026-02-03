#!/usr/bin/env python3
"""
Workflow completo per creare mazzo di carte quantitative:
1. Web discovery con LLM filtering
2. Review e raffinamento con LLM
3. Download immagini
4. Analisi quantitativa
"""

import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from web_discovery import discover_entities
from llm_analyzer import get_llm_config, call_llm


def step1_discover():
    """Step 1: Web discovery con Brave Search"""
    print("\n" + "=" * 60)
    print("  STEP 1: Web Discovery - Cerca informatici famosi")
    print("=" * 60)

    query = "famous computer scientists history pioneers"
    print(f"\n🔍 Cerco: '{query}'")

    try:
        entities = discover_entities(query=query, entity_type="person", max_results=20)
    except Exception as e:
        print(f"❌ Errore nel discovery: {e}")
        return None

    if not entities:
        print("❌ Nessuna entità trovata")
        return None

    print(f"✓ Trovate {len(entities)} potenziali entità")

    return entities


def step2_llm_filter(entities):
    """Step 2: Filtra con LLM per tenere solo informatici veri"""
    print("\n" + "=" * 60)
    print("  STEP 2: Filtraggio LLM - Solo informatici veri")
    print("=" * 60)

    print("\n🤖 Uso LLM per filtrare...")

    try:
        config = get_llm_config("auto")
    except ValueError:
        print("❌ Nessun LLM disponibile")
        return entities

    prompt = f"""You are filtering a list of entities to keep only real computer scientists, software engineers, and technology pioneers.

Entities found:
{chr(10).join(f"- {e}" for e in entities)}

Filter this list to keep ONLY:
- Real people (not organizations, concepts, or places)
- Computer scientists, software engineers, mathematicians who contributed to computing
- Technology pioneers and innovators

Remove:
- Organizations (e.g., "Amazon Web Services", "Microsoft")
- Concepts (e.g., "Artificial Intelligence", "Machine Learning")
- Generic phrases (e.g., "Famous Computer Scientists")
- Places or universities

Return ONLY a JSON array of person names that should be kept.
Example format: ["Alan Turing", "Grace Hopper", "Linus Torvalds"]

JSON array:"""

    response = call_llm(config=config, prompt=prompt)

    # Parse JSON
    import re

    json_match = re.search(r"\[.*\]", response, re.DOTALL)
    if json_match:
        filtered = json.loads(json_match.group(0))
        print(f"✓ LLM ha filtrato: {len(entities)} → {len(filtered)} informatici")
        return filtered

    print("⚠️  LLM response non parsabile, uso lista originale")
    return entities


def step3_llm_expand(scientists):
    """Step 3: Espandi la lista con altri informatici famosi suggeriti da LLM"""
    print("\n" + "=" * 60)
    print("  STEP 3: Espansione LLM - Aggiungi altri famosi")
    print("=" * 60)

    print(f"\n📋 Lista attuale: {len(scientists)} informatici")
    print("\n🤖 Chiedo a LLM di suggerire altri informatici importanti...")

    try:
        config = get_llm_config("auto")
    except ValueError:
        print("❌ Nessun LLM disponibile")
        return scientists

    current_list = "\n".join(f"- {s}" for s in scientists)

    prompt = f"""Current list of computer scientists for a trading card game:
{current_list}

This list seems incomplete. Suggest 10-15 MORE legendary computer scientists who should be in this collection.

Include a diverse mix:
- Historical pioneers (1940s-1980s) like Turing, Hopper
- Modern innovators (1990s-2020s) like Torvalds, van Rossum
- Different domains: algorithms, languages, systems, AI, web, security
- Diverse backgrounds and genders

Exclude anyone already in the current list.

Return ONLY a JSON array of additional names to add.
Example: ["Donald Knuth", "Grace Hopper", "Dennis Ritchie"]

JSON array:"""

    response = call_llm(config=config, prompt=prompt)

    # Parse JSON
    import re

    json_match = re.search(r"\[.*\]", response, re.DOTALL)
    if json_match:
        additional = json.loads(json_match.group(0))
        print(f"✓ LLM suggerisce {len(additional)} informatici aggiuntivi:")
        for name in additional[:10]:
            print(f"   + {name}")
        if len(additional) > 10:
            print(f"   ... e altri {len(additional) - 10}")

        combined = list(set(scientists + additional))
        print(f"\n✓ Lista totale: {len(combined)} informatici")
        return combined

    print("⚠️  LLM response non parsabile, uso lista attuale")
    return scientists


def step4_user_review(scientists, auto_accept=False):
    """Step 4: Review manuale da parte dell'utente"""
    print("\n" + "=" * 60)
    print("  STEP 4: Review Finale")
    print("=" * 60)

    print(f"\n📋 Lista finale proposta ({len(scientists)} informatici):\n")

    for i, name in enumerate(sorted(scientists), 1):
        print(f"  {i:2}. {name}")

    if auto_accept:
        print("\n✓ Auto-accettazione attiva")
        return scientists

    print("\n" + "=" * 60)
    print("Vuoi procedere con questa lista?")
    print("  [y] Si, procedi")
    print("  [e] Modifica (rimuovi alcuni nomi)")
    print("  [a] Aggiungi altri nomi")
    print("  [n] Ricomincia")
    print("=" * 60)

    try:
        choice = input("\nScelta: ").strip().lower()
    except EOFError:
        print("\n✓ Non-interactive mode: auto-accepting list")
        return scientists

    if choice == "y":
        return scientists
    elif choice == "e":
        print("\nInserisci i numeri da RIMUOVERE (separati da virgola):")
        to_remove = input("Numeri: ").strip()
        if to_remove:
            indices = [int(x.strip()) - 1 for x in to_remove.split(",")]
            sorted_list = sorted(scientists)
            scientists = [s for i, s in enumerate(sorted_list) if i not in indices]
            print(f"✓ Rimossi. Nuova lista: {len(scientists)} informatici")
        return scientists
    elif choice == "a":
        print("\nInserisci nomi da aggiungere (uno per riga, invio vuoto per finire):")
        while True:
            name = input("Nome: ").strip()
            if not name:
                break
            scientists.append(name)
            print(f"  + Aggiunto: {name}")
        return scientists
    else:
        return None


def save_list(scientists, filename="informatici.txt"):
    """Salva la lista in formato txt"""
    with open(filename, "w") as f:
        for name in sorted(scientists):
            f.write(f"{name}, computer scientist\n")
    print(f"\n✅ Lista salvata in: {filename}")
    print(f"   Totale: {len(scientists)} informatici")


if __name__ == "__main__":
    print("\n🎮 NIMITZ - Creazione Mazzo Informatici")
    print("=" * 60)

    # Step 1: Web Discovery
    entities = step1_discover()
    if not entities:
        print("\n❌ Errore nel discovery")
        sys.exit(1)

    # Step 2: LLM Filter
    scientists = step2_llm_filter(entities)

    # Step 3: LLM Expand
    scientists = step3_llm_expand(scientists)

    # Step 4: User Review
    final_list = step4_user_review(scientists)

    if final_list:
        save_list(final_list, "informatici.txt")
        print("\n" + "=" * 60)
        print("✅ STEP 1 COMPLETATO!")
        print("\nProssimi passi:")
        print(
            "  2. Scarica immagini: nimitz retrieve batch informatici.txt --source pexels -o ./informatici_cards"
        )
        print("  3. Crea vocabolario: nimitz wizard")
        print("  4. Analizza: python analyze_quantitative.py")
        print("=" * 60)
    else:
        print("\n❌ Processo annullato")
