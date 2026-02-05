#!/usr/bin/env python3
"""
Step 3: Enriched Analysis - Web Research + CLIP
Combina ricerca web e analisi CLIP per generare valori accurati delle card
"""

import sys
import json
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

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

from quantitative_scoring import convert_to_quantitative_scores
from llm_analyzer import get_llm_config, call_llm
from web_discovery import BraveSearchClient


def load_vocabulary(vocab_file: str) -> Dict:
    """Carica vocabolario JSON"""
    with open(vocab_file) as f:
        return json.load(f)


def get_scientist_images(cards_dir: str) -> List[Tuple[str, str]]:
    """
    Trova tutte le immagini degli informatici
    Returns: List of (name, image_path)
    """
    images = []
    for img_path in glob.glob(f"{cards_dir}/*.jpeg"):
        filename = Path(img_path).stem
        # Converti nome file in nome leggibile
        # "Ada_Lovelace__computer_scientist" -> "Ada Lovelace"
        name = filename.split("__")[0].replace("_", " ")
        images.append((name, img_path))

    return sorted(images)


def web_research_scientist(
    name: str, brave_client: Optional[BraveSearchClient] = None
) -> Dict:
    """
    Ricerca informazioni su un informatico dal web

    Args:
        name: Nome dell'informatico
        brave_client: Client Brave Search (opzionale)

    Returns:
        Dict con informazioni trovate online
    """
    if not brave_client:
        return {"raw_data": None, "summary": "No web research available"}

    try:
        # Cerca informazioni biografiche
        query = f'"{name}" computer scientist biography achievements contributions'
        results = brave_client.search(query, count=5)

        # Estrai testo dai risultati
        web_results = results.get("web", {}).get("results", [])

        snippets = []
        for result in web_results[:3]:  # Top 3 risultati
            title = result.get("title", "")
            description = result.get("description", "")
            snippets.append(f"{title}\n{description}")

        combined_text = "\n\n".join(snippets)

        return {
            "raw_data": combined_text,
            "summary": f"Found {len(snippets)} web sources",
        }

    except Exception as e:
        print(f"   ⚠️  Web research failed: {e}")
        return {"raw_data": None, "summary": "Web research failed"}


def analyze_with_clip(
    image_path: str, prompts: List[str], clip_model_tuple=None
) -> List[float]:
    """
    Analizza immagine con CLIP
    Returns: List of similarity scores (0-1) per ogni prompt
    """
    try:
        from PIL import Image
        import torch
        import clip

        # Load model if not provided
        if clip_model_tuple is None:
            from embed import initialize_clip_model

            model, preprocess, device = initialize_clip_model()
        else:
            model, preprocess, device = clip_model_tuple

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Tokenize all prompts at once
        text_tokens = clip.tokenize(prompts).to(device)

        # Extract features
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_tokens)

            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Compute similarities (cosine similarity)
            similarities = (image_features @ text_features.T).squeeze(0)

        # Convert to Python list
        similarities = similarities.cpu().numpy().tolist()

        return similarities

    except Exception as e:
        print(f"   ⚠️  CLIP analysis failed: {e}")
        print("      Using fallback: simulated scores")
        import random

        # Use weighted random to simulate progression
        return [
            random.uniform(0.1 + i * 0.1, 0.25 + i * 0.15) for i in range(len(prompts))
        ]


def llm_score_from_web_data(
    name: str, characteristic_name: str, prompts: List[str], web_data: Dict, llm_config
) -> Optional[int]:
    """
    Usa LLM per estrarre uno score da dati web

    Args:
        name: Nome dell'informatico
        characteristic_name: Nome della caratteristica (es. "influenza_sul_mercato")
        prompts: Lista dei 5 prompt (LOW -> HIGH)
        web_data: Dati web raccolti
        llm_config: Configurazione LLM

    Returns:
        Score 0-100, o None se non disponibile
    """
    if not web_data.get("raw_data"):
        return None

    try:
        # Prepara prompt per LLM
        char_display = characteristic_name.replace("_", " ").title()

        prompt = f"""You are analyzing {name}, a computer scientist.

Based on the following web research data, rate their {char_display} on a scale of 0-100.

Web Data:
{web_data["raw_data"][:2000]}  

Rating Scale (use these as reference):
0-20: {prompts[0]}
20-40: {prompts[1]}
40-60: {prompts[2]}
60-80: {prompts[3]}
80-100: {prompts[4]}

Consider:
- Historical impact and legacy
- Current relevance and influence
- Scope of contributions
- Recognition by peers and industry

Return ONLY a number between 0-100, nothing else.
Score:"""

        response = call_llm(config=llm_config, prompt=prompt)

        # Estrai numero dalla risposta
        match = re.search(r"\b(\d{1,3})\b", response)
        if match:
            score = int(match.group(1))
            if 0 <= score <= 100:
                return score

        return None

    except Exception as e:
        print(f"      ⚠️  LLM scoring failed: {e}")
        return None


def analyze_scientist_enriched(
    name: str,
    image_path: str,
    characteristics: Dict,
    clip_model,
    brave_client: Optional[BraveSearchClient],
    llm_config,
    use_web: bool = True,
    use_llm_scoring: bool = True,
) -> Dict:
    """
    Analizza uno scienziato combinando CLIP + Web + LLM

    Args:
        name: Nome dello scienziato
        image_path: Path all'immagine
        characteristics: Dizionario delle caratteristiche dal vocabolario
        clip_model: Modello CLIP inizializzato
        brave_client: Client Brave Search
        llm_config: Configurazione LLM
        use_web: Se True, usa web research per arricchire i dati
        use_llm_scoring: Se True, usa LLM per scoring da web data

    Returns:
        Dict con dati della card
    """
    card_data = {"name": name, "image": image_path, "scores": {}, "sources": {}}

    # Web Research (se abilitato)
    web_data = None
    if use_web and brave_client:
        print(f"   🌐 Web research...")
        web_data = web_research_scientist(name, brave_client)
        print(f"      {web_data['summary']}")

    # Analizza ogni caratteristica
    for char_name, prompts in characteristics.items():
        display_name = char_name.replace("_", " ").title()

        # METODO 1: LLM scoring da web data (se disponibile)
        llm_score = None
        if use_llm_scoring and web_data and web_data.get("raw_data"):
            print(f"   🤖 LLM scoring: {display_name}...")
            llm_score = llm_score_from_web_data(
                name, char_name, prompts, web_data, llm_config
            )
            if llm_score is not None:
                print(f"      LLM score: {llm_score}/100")

        # METODO 2: CLIP analysis dall'immagine
        print(f"   🖼️  CLIP analysis: {display_name}...")
        similarities_list = analyze_with_clip(image_path, prompts, clip_model)

        import numpy as np

        similarities = np.array(similarities_list)

        clip_result = convert_to_quantitative_scores(
            similarities=similarities,
            characteristic_name=char_name,
            prompts=prompts,
        )
        clip_score = int(clip_result["score"])
        print(f"      CLIP score: {clip_score}/100")

        # COMBINAZIONE: Media pesata LLM (70%) + CLIP (30%)
        # Se LLM non disponibile, usa solo CLIP
        if llm_score is not None:
            final_score = int(llm_score * 0.7 + clip_score * 0.3)
            source = "LLM+CLIP"
        else:
            final_score = clip_score
            source = "CLIP_only"

        card_data["scores"][char_name] = final_score
        card_data["sources"][char_name] = source

        print(f"      ✓ Final: {final_score}/100 (from {source})")

    return card_data


def create_card_display(
    name: str, scores: Dict[str, int], rank: int, total: int
) -> str:
    """Crea rappresentazione ASCII della carta"""

    # Trova caratteristica più alta
    max_char = max(scores.items(), key=lambda x: x[1])

    # Crea carta
    card = f"""
╔══════════════════════════════════════════╗
║  #{rank}/{total}  {"⭐" * min(5, rank)}
║
║  {name[:38].center(38)}
║
╠══════════════════════════════════════════╣
"""

    # Aggiungi caratteristiche
    for char_name, score in scores.items():
        # Crea barra di progresso
        bar_length = 20
        filled = int(score / 100 * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)

        # Evidenzia la migliore
        prefix = "🏆" if char_name == max_char[0] else "  "

        # Nome caratteristica formattato
        display_name = char_name.replace("_", " ").title()[:20]

        card += f"║ {prefix} {display_name:<20} {score:3d} {bar} ║\n"

    card += "╚══════════════════════════════════════════╝"

    return card


def main():
    print("\n" + "=" * 70)
    print("  NIMITZ - Analisi Arricchita (Web + CLIP + LLM)")
    print("=" * 70)

    # Load vocabulary
    vocab_file = "vocabolario_informatici_game.json"
    print(f"\n📖 Caricamento vocabolario: {vocab_file}")
    vocab = load_vocabulary(vocab_file)

    characteristics = vocab["characteristics"]
    print(f"✓ Caricate {len(characteristics)} caratteristiche")

    # Find images
    cards_dir = "./informatici_cards"
    print(f"\n🖼️  Cerca immagini in: {cards_dir}")
    scientists = get_scientist_images(cards_dir)
    print(f"✓ Trovate {len(scientists)} immagini")

    # Initialize CLIP model
    print("\n🤖 Inizializzazione CLIP model...")
    try:
        from embed import initialize_clip_model

        clip_model = initialize_clip_model()
        print("✓ CLIP model caricato")
    except Exception as e:
        print(f"⚠️  CLIP non disponibile: {e}")
        print("   Uso fallback con punteggi simulati")
        clip_model = None

    # Initialize Brave Search
    print("\n🌐 Inizializzazione Brave Search...")
    brave_client = None
    try:
        import os

        if os.getenv("BRAVE_API_KEY"):
            brave_client = BraveSearchClient()
            print("✓ Brave Search API configurata")
        else:
            print("⚠️  BRAVE_API_KEY non trovata")
            print("   Continuo senza web research")
    except Exception as e:
        print(f"⚠️  Brave Search non disponibile: {e}")
        print("   Continuo senza web research")

    # Initialize LLM
    print("\n🧠 Inizializzazione LLM...")
    llm_config = None
    try:
        llm_config = get_llm_config("auto")
        print(f"✓ LLM configurato: {llm_config.model}")
    except Exception as e:
        print(f"⚠️  LLM non disponibile: {e}")
        print("   Continuo solo con CLIP")

    # Configuration
    use_web = brave_client is not None
    use_llm_scoring = llm_config is not None and use_web

    print("\n" + "=" * 70)
    print("  CONFIGURAZIONE")
    print("=" * 70)
    print(
        f"  CLIP Analysis:      {'✓ Enabled' if clip_model else '✗ Disabled (fallback)'}"
    )
    print(f"  Web Research:       {'✓ Enabled' if use_web else '✗ Disabled'}")
    print(f"  LLM Scoring:        {'✓ Enabled' if use_llm_scoring else '✗ Disabled'}")
    print("=" * 70)

    # Analyze each scientist
    print("\n🔬 Analisi in corso...\n")

    all_cards = []

    for i, (name, img_path) in enumerate(scientists, 1):
        print(f"\n[{i}/{len(scientists)}] {'=' * 60}")
        print(f"  Analizzo: {name}")
        print("=" * 60)

        card_data = analyze_scientist_enriched(
            name=name,
            image_path=img_path,
            characteristics=characteristics,
            clip_model=clip_model,
            brave_client=brave_client,
            llm_config=llm_config,
            use_web=use_web,
            use_llm_scoring=use_llm_scoring,
        )

        all_cards.append(card_data)

    # Calculate rankings
    print("\n\n" + "=" * 70)
    print("  CLASSIFICA TOP 3 PER CARATTERISTICA")
    print("=" * 70 + "\n")

    for char_name in characteristics.keys():
        display_name = char_name.replace("_", " ").title()
        print(f"\n🏆 {display_name.upper()}")
        print("-" * 70)

        # Sort by this characteristic
        ranked = sorted(all_cards, key=lambda x: x["scores"][char_name], reverse=True)

        for rank, card in enumerate(ranked[:3], 1):
            medal = ["🥇", "🥈", "🥉"][rank - 1]
            score = card["scores"][char_name]
            source = card["sources"][char_name]
            print(f"  {medal} {rank}. {card['name']:<30} {score:3d}/100 ({source})")

    # Display sample cards
    print("\n" + "=" * 70)
    print("  TOP 5 CARTE (per punteggio medio)")
    print("=" * 70)

    # Calculate average scores
    for card in all_cards:
        card["avg_score"] = sum(card["scores"].values()) / len(card["scores"])

    ranked_overall = sorted(all_cards, key=lambda x: x["avg_score"], reverse=True)

    for i, card in enumerate(ranked_overall[:5], 1):
        print(
            create_card_display(
                name=card["name"], scores=card["scores"], rank=i, total=len(all_cards)
            )
        )
        print()

    # Save results
    output_file = "informatici_cards_analysis.json"

    # Remove "sources" from saved data (internal info)
    save_data = []
    for card in all_cards:
        card_copy = card.copy()
        card_copy.pop("sources", None)
        card_copy.pop("avg_score", None)
        save_data.append(card_copy)

    with open(output_file, "w") as f:
        json.dump(save_data, f, indent=2)

    print("\n" + "=" * 70)
    print(f"✅ Analisi completata!")
    print(f"   Risultati salvati in: {output_file}")
    print(f"   Totale carte: {len(all_cards)}")

    # Statistics
    web_enhanced = sum(
        1 for c in all_cards if any("LLM" in v for v in c.get("sources", {}).values())
    )
    print(f"\n📊 Statistiche:")
    print(f"   Carte con web research: {web_enhanced}/{len(all_cards)}")
    print(f"   Carte solo CLIP: {len(all_cards) - web_enhanced}/{len(all_cards)}")

    print("=" * 70)

    # Generate visual cards
    print("\n" + "=" * 70)
    print("  GENERAZIONE CARD GRAFICHE")
    print("=" * 70)

    try:
        from image_card import create_visual_image_cards

        # Convert enriched data format to image_card format
        cards_data_visual = []
        for card in all_cards:
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

            cards_data_visual.append(card_visual)

        # Create visual cards
        output_cards_dir = "./informatici_cards_visual"
        create_visual_image_cards(
            cards_data=cards_data_visual,
            output_dir=output_cards_dir,
            cards_per_page=6,
            show_thumbnails=True,
        )

        print(f"\n✅ Card grafiche generate in: {output_cards_dir}")
        print(f"   - Pagine multiple: {output_cards_dir}/image_cards_page_*.png")
        print(f"   - Card individuali: {output_cards_dir}/individual/")

    except ImportError as e:
        print(f"\n⚠️  Impossibile generare card grafiche: modulo mancante")
        print(f"   Errore: {e}")
        print(f"   Installa matplotlib e pillow: pip install matplotlib pillow")
    except Exception as e:
        print(f"\n⚠️  Errore durante generazione card grafiche: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
