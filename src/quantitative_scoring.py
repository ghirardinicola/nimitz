#!/usr/bin/env python3
"""
Converte le similarità CLIP in punteggi quantitativi (0-100)
per creare carte da gioco con stats numeriche.
"""

import numpy as np
from typing import Dict, List


def convert_to_quantitative_scores(
    similarities: np.ndarray, characteristic_name: str, prompts: List[str]
) -> Dict[str, float]:
    """
    Converte similarità CLIP in punteggio quantitativo.

    Assumption: i prompt sono ordinati da LOW a HIGH

    Args:
        similarities: Array di similarità per ogni prompt (0-1)
        characteristic_name: Nome della caratteristica
        prompts: Lista di prompt ordinati dal più basso al più alto

    Returns:
        Dict con punteggio 0-100
    """
    # Weighted sum basato sulla posizione del prompt
    # Prompt più alto (ultimo) = peso maggiore
    num_prompts = len(prompts)
    weights = np.arange(num_prompts) / (num_prompts - 1)  # 0, 0.25, 0.5, 0.75, 1.0

    # Score = weighted average delle similarità
    weighted_score = np.sum(similarities * weights) / np.sum(similarities)

    # Converti in scala 0-100
    score = weighted_score * 100

    return {
        "score": round(score, 1),
        "best_match_index": int(np.argmax(similarities)),
        "best_match_prompt": prompts[int(np.argmax(similarities))],
        "confidence": round(float(np.max(similarities)), 3),
    }


# Test
if __name__ == "__main__":
    # Simula similarità CLIP per "impatto_storico"
    prompts = [
        "contributore modesto a progetti informatici locali",  # 20/100
        "sviluppatore di software utilizzato da poche persone",  # 40/100
        "creatore di strumenti utilizzati da una comunità",  # 60/100
        "programmatore che ha influenzato un'intera industria",  # 80/100
        "visionario che ha plasmato l'intero settore informatico",  # 100/100
    ]

    # Test case 1: Immagine matcha con "visionario" (Alan Turing)
    similarities_turing = np.array([0.15, 0.18, 0.22, 0.25, 0.35])
    result = convert_to_quantitative_scores(
        similarities_turing, "impatto_storico", prompts
    )
    print("Alan Turing - Impatto Storico:")
    print(f"  Score: {result['score']}/100")
    print(f"  Best match: {result['best_match_prompt']}")
    print()

    # Test case 2: Programmatore medio
    similarities_avg = np.array([0.20, 0.25, 0.30, 0.18, 0.15])
    result = convert_to_quantitative_scores(
        similarities_avg, "impatto_storico", prompts
    )
    print("Programmatore Medio - Impatto Storico:")
    print(f"  Score: {result['score']}/100")
    print(f"  Best match: {result['best_match_prompt']}")
