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
import os
from pathlib import Path

# Load .env file if exists
env_file = Path(".env")
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value

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


def step3_llm_expand(scientists, target_count=32):
    """Step 3: Espandi la lista con altri informatici famosi suggeriti da LLM"""
    print("\n" + "=" * 60)
    print("  STEP 3: Espansione LLM - Obiettivo 32 carte")
    print("=" * 60)

    print(f"\n📋 Lista attuale: {len(scientists)} informatici")
    print(f"🎯 Target: {target_count} carte (come un mazzo Top Trumps classico)")

    needed = target_count - len(scientists)
    if needed <= 0:
        print(f"✓ Hai già {len(scientists)} informatici, sufficiente!")
        return scientists

    print(f"\n🤖 Chiedo a LLM di suggerire {needed} informatici aggiuntivi...")

    try:
        config = get_llm_config("auto")
    except ValueError:
        print("❌ Nessun LLM disponibile")
        return scientists

    current_list = "\n".join(f"- {s}" for s in scientists)

    prompt = f"""Current list of computer scientists for a Top Trumps trading card game (target: 32 cards):
{current_list}

We need {needed} MORE legendary computer scientists to complete the deck (32 total).

Suggest exactly {needed} additional names with these criteria:
- Historical pioneers (1940s-1980s): Turing, Hopper, Knuth, Dijkstra
- Modern innovators (1990s-2020s): Torvalds, van Rossum, Berners-Lee
- Different domains: algorithms, languages, systems, AI, web, security, cryptography
- Diverse backgrounds and genders
- Only REAL people who are famous and recognizable

Exclude anyone already in the current list above.

Return ONLY a JSON array of exactly {needed} names.
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

        if len(combined) > target_count:
            print(f"⚠️  Abbiamo {len(combined)} nomi (target: {target_count})")
            print(
                f"   Potrai selezionare i migliori {target_count} nello step successivo"
            )

        return combined

    print("⚠️  LLM response non parsabile, uso lista attuale")
    return scientists


def step4_user_review(scientists, target_count=32, auto_accept=False):
    """Step 4: Review interattivo con TUI per selezione"""
    print("\n" + "=" * 60)
    print(f"  STEP 4: Selezione Finale - Target {target_count} carte")
    print("=" * 60)

    print(f"\n📋 Candidati disponibili: {len(scientists)} informatici")

    if auto_accept:
        print("\n✓ Auto-accettazione attiva")
        return scientists[:target_count]

    # Se abbiamo già il numero giusto, mostra semplice conferma
    if len(scientists) == target_count:
        print(f"\n✓ Perfetto! Hai esattamente {target_count} informatici.")
        sorted_scientists = sorted(scientists)
        for i, name in enumerate(sorted_scientists, 1):
            print(f"  {i:2}. {name}")

        print("\n" + "=" * 60)
        try:
            choice = input("Procedere con questa lista? [y/n]: ").strip().lower()
        except EOFError:
            return scientists

        if choice == "y":
            return scientists
        # Altrimenti continua con selezione interattiva sotto

    # TUI Selection
    print(f"\n🎯 Seleziona esattamente {target_count} informatici per il tuo mazzo")
    print("=" * 60)
    print("Controlli:")
    print("  [SPACE] Seleziona/Deseleziona")
    print("  [ENTER] Conferma selezione")
    print("  [a]     Aggiungi nuovo nome")
    print("  [q]     Annulla")
    print("=" * 60)

    try:
        selected = interactive_selection_tui(scientists, target_count)
        return selected
    except KeyboardInterrupt:
        print("\n\n❌ Selezione annullata")
        return None
    except EOFError:
        # Non-interactive mode
        print(f"\n✓ Modalità non-interattiva: seleziono i primi {target_count}")
        return sorted(scientists)[:target_count]


def interactive_selection_tui(scientists, target_count):
    """TUI interattiva per selezione con spazio"""
    import sys
    import tty
    import termios

    sorted_scientists = sorted(scientists)
    selected = set()
    current_idx = 0
    scroll_offset = 0

    # Pre-seleziona i primi target_count se ce ne sono meno
    if len(sorted_scientists) <= target_count:
        selected = set(range(len(sorted_scientists)))

    def get_key():
        """Legge un singolo carattere da stdin"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            # Handle special keys
            if ch == "\x1b":  # ESC sequence
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    if ch3 == "A":
                        return "UP"
                    elif ch3 == "B":
                        return "DOWN"
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def draw_screen():
        """Disegna la schermata di selezione"""
        nonlocal scroll_offset

        # Clear screen
        print("\033[2J\033[H", end="")

        print("╔" + "═" * 78 + "╗")
        print(
            f"║ 🎮 SELEZIONE INFORMATICI - {len(selected)}/{target_count} selezionati"
            + " " * (78 - 45 - len(str(len(selected))) - len(str(target_count)))
            + "║"
        )
        print("╠" + "═" * 78 + "╣")

        # Visible window (20 items)
        visible_count = 20
        max_scroll = max(0, len(sorted_scientists) - visible_count)

        # Auto-scroll to keep current item visible
        if current_idx < scroll_offset:
            scroll_offset = current_idx
        elif current_idx >= scroll_offset + visible_count:
            scroll_offset = current_idx - visible_count + 1

        scroll_offset = min(scroll_offset, max_scroll)

        for i in range(
            scroll_offset, min(scroll_offset + visible_count, len(sorted_scientists))
        ):
            name = sorted_scientists[i]

            # Indicators
            cursor = "→" if i == current_idx else " "
            checkbox = "☑" if i in selected else "☐"

            # Truncate name if too long
            display_name = name[:65]

            line = f"║ {cursor} {checkbox} {i + 1:2}. {display_name}"
            padding = 78 - len(line) + 1
            print(line + " " * padding + "║")

        # Fill remaining lines
        for _ in range(
            visible_count - min(visible_count, len(sorted_scientists) - scroll_offset)
        ):
            print("║" + " " * 78 + "║")

        print("╠" + "═" * 78 + "╣")
        print(
            f"║ [SPACE] Seleziona  [↑↓] Naviga  [ENTER] Conferma  [a] Aggiungi  [q] Esci  ║"
        )
        print("╚" + "═" * 78 + "╝")

        # Status message
        if len(selected) > target_count:
            print(
                f"\n⚠️  Hai selezionato {len(selected)} carte (troppi! target: {target_count})"
            )
        elif len(selected) < target_count:
            print(f"\n📝 Seleziona altre {target_count - len(selected)} carte")
        else:
            print(
                f"\n✅ Perfetto! Hai selezionato {target_count} carte. Premi ENTER per confermare."
            )

    def add_custom_name():
        """Aggiungi un nome custom"""
        # Restore terminal
        print("\033[2J\033[H")
        print("╔" + "═" * 78 + "╗")
        print("║ Aggiungi Nuovo Informatico" + " " * 51 + "║")
        print("╚" + "═" * 78 + "╝")
        print()

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        try:
            name = input("Nome completo (ENTER per annullare): ").strip()
            if name:
                sorted_scientists.append(name)
                sorted_scientists.sort()
                print(f"✓ Aggiunto: {name}")
                import time

                time.sleep(0.5)
                return True
        except:
            pass

        return False

    # Main loop
    while True:
        draw_screen()

        key = get_key()

        if key == "UP" or key == "k":
            current_idx = max(0, current_idx - 1)
        elif key == "DOWN" or key == "j":
            current_idx = min(len(sorted_scientists) - 1, current_idx + 1)
        elif key == " ":  # Space
            if current_idx in selected:
                selected.remove(current_idx)
            else:
                selected.add(current_idx)
        elif key == "\n" or key == "\r":  # Enter
            if len(selected) == target_count:
                break
            elif len(selected) < target_count:
                # Warn but allow
                print(
                    f"\n⚠️  Hai solo {len(selected)}/{target_count} carte selezionate."
                )
                print("Confermi comunque? [y/n]: ", end="", flush=True)

                # Restore terminal for input
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

                confirm = input().strip().lower()
                if confirm == "y":
                    break
            else:
                print(f"\n⚠️  Hai {len(selected)} carte (troppi! max: {target_count})")
                print("Premi un tasto per continuare...", end="", flush=True)
                get_key()
        elif key == "a" or key == "A":
            if add_custom_name():
                # Resort and update indices
                new_selected = set()
                selected_names = {
                    sorted_scientists[i] for i in selected if i < len(sorted_scientists)
                }
                for i, name in enumerate(sorted_scientists):
                    if name in selected_names:
                        new_selected.add(i)
                selected = new_selected
        elif key == "q" or key == "Q":
            print("\033[2J\033[H")
            raise KeyboardInterrupt

    # Return selected scientists
    print("\033[2J\033[H")
    result = [sorted_scientists[i] for i in sorted(selected)]

    print("✅ Selezione confermata!")
    print(f"\n{len(result)} informatici selezionati:\n")
    for i, name in enumerate(result, 1):
        print(f"  {i:2}. {name}")

    return result


def save_list(scientists, filename="informatici.txt"):
    """Salva la lista in formato txt"""
    with open(filename, "w") as f:
        for name in sorted(scientists):
            f.write(f"{name}, computer scientist\n")
    print(f"\n✅ Lista salvata in: {filename}")
    print(f"   Totale: {len(scientists)} informatici")


if __name__ == "__main__":
    print("\n🎮 NIMITZ - Creazione Mazzo Informatici (32 carte)")
    print("=" * 60)

    TARGET_CARDS = 32  # Come un mazzo Top Trumps classico

    # Step 1: Web Discovery
    entities = step1_discover()
    if not entities:
        print("\n❌ Errore nel discovery")
        sys.exit(1)

    # Step 2: LLM Filter
    scientists = step2_llm_filter(entities)

    # Step 3: LLM Expand (fino a ~40-50 per avere scelta)
    scientists = step3_llm_expand(scientists, target_count=TARGET_CARDS)

    # Step 4: User Review (selezione interattiva di 32)
    final_list = step4_user_review(scientists, target_count=TARGET_CARDS)

    if final_list:
        save_list(final_list, "informatici.txt")
        print("\n" + "=" * 60)
        print("✅ STEP 1 COMPLETATO!")
        print(
            f"\n🎯 {len(final_list)} informatici selezionati (target: {TARGET_CARDS})"
        )
        print("\nProssimi passi:")
        print(
            "  2. Scarica immagini: nimitz retrieve batch informatici.txt --source pexels -o ./informatici_cards"
        )
        print("  3. Crea vocabolario: nimitz wizard")
        print("  4. Analizza: python create_deck_step3_enriched.py")
        print("=" * 60)
    else:
        print("\n❌ Processo annullato")
