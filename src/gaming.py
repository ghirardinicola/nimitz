#!/usr/bin/env python3
"""
NIMITZ - Gaming Module
Card battles, rarity calculation, and power levels
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from enum import Enum


class RarityTier(Enum):
    """Card rarity tiers based on feature uniqueness"""
    COMMON = "common"
    UNCOMMON = "uncommon"
    RARE = "rare"
    EPIC = "epic"
    LEGENDARY = "legendary"


# Rarity thresholds (percentile-based)
RARITY_THRESHOLDS = {
    RarityTier.LEGENDARY: 95,   # Top 5%
    RarityTier.EPIC: 85,        # Top 15%
    RarityTier.RARE: 70,        # Top 30%
    RarityTier.UNCOMMON: 50,    # Top 50%
    RarityTier.COMMON: 0        # Bottom 50%
}

RARITY_COLORS = {
    RarityTier.COMMON: "#9E9E9E",      # Gray
    RarityTier.UNCOMMON: "#4CAF50",    # Green
    RarityTier.RARE: "#2196F3",        # Blue
    RarityTier.EPIC: "#9C27B0",        # Purple
    RarityTier.LEGENDARY: "#FF9800"    # Orange/Gold
}

RARITY_SYMBOLS = {
    RarityTier.COMMON: "",
    RarityTier.UNCOMMON: "",
    RarityTier.RARE: "",
    RarityTier.EPIC: "",
    RarityTier.LEGENDARY: ""
}


def calculate_power_level(card: Dict[str, Any]) -> float:
    """
    Calculate overall power level of a card.

    Power level is a weighted combination of:
    - Overall max score (40%)
    - Mean score (30%)
    - High confidence features count (30%)

    Args:
        card: Card dictionary with feature_summary

    Returns:
        Power level between 0 and 100
    """
    summary = card.get('feature_summary', {})

    max_score = summary.get('overall_max', 0)
    mean_score = summary.get('overall_mean', 0)
    high_conf = summary.get('high_confidence_features', 0)
    total_features = summary.get('total_features', 1)

    # Normalize high confidence ratio
    high_conf_ratio = high_conf / max(total_features, 1)

    # Calculate weighted power level
    power = (max_score * 0.4 + mean_score * 0.3 + high_conf_ratio * 0.3) * 100

    return round(power, 2)


def calculate_rarity_score(
    card: Dict[str, Any],
    all_cards: List[Dict[str, Any]]
) -> Tuple[float, RarityTier]:
    """
    Calculate rarity score based on how unique the card's features are.

    Rarity is based on:
    - How different the card's feature vector is from others
    - Presence of unusual high-scoring characteristics
    - Unique combinations of dominant features

    Args:
        card: The card to evaluate
        all_cards: All cards in the collection for comparison

    Returns:
        Tuple of (rarity_score, rarity_tier)
    """
    if len(all_cards) <= 1:
        return 50.0, RarityTier.UNCOMMON

    # Extract feature vector for the target card
    target_features = _extract_feature_vector(card)

    if target_features is None:
        return 50.0, RarityTier.UNCOMMON

    # Calculate distances to all other cards
    distances = []
    for other_card in all_cards:
        if other_card.get('image_name') == card.get('image_name'):
            continue

        other_features = _extract_feature_vector(other_card)
        if other_features is not None:
            # Euclidean distance
            dist = np.linalg.norm(target_features - other_features)
            distances.append(dist)

    if not distances:
        return 50.0, RarityTier.UNCOMMON

    # Average distance to other cards
    avg_distance = np.mean(distances)

    # Normalize to 0-100 scale (higher = more unique)
    # Using sigmoid-like transformation
    max_possible_distance = np.sqrt(len(target_features))  # Max Euclidean distance
    rarity_score = min(100, (avg_distance / max_possible_distance) * 150)

    # Boost for high max scores (exceptional cards are rarer)
    max_score = card.get('feature_summary', {}).get('overall_max', 0)
    if max_score > 0.9:
        rarity_score = min(100, rarity_score * 1.2)
    elif max_score > 0.8:
        rarity_score = min(100, rarity_score * 1.1)

    # Determine tier based on score
    tier = _score_to_tier(rarity_score)

    return round(rarity_score, 2), tier


def _extract_feature_vector(card: Dict[str, Any]) -> Optional[np.ndarray]:
    """Extract a flat feature vector from card characteristics"""
    characteristics = card.get('characteristics', {})

    if not characteristics:
        return None

    features = []
    for char_name in sorted(characteristics.keys()):
        char_info = characteristics[char_name]
        if 'scores' in char_info:
            features.extend(char_info['scores'])
        elif 'max_score' in char_info:
            features.append(char_info['max_score'])

    return np.array(features) if features else None


def _score_to_tier(score: float) -> RarityTier:
    """Convert rarity score to tier"""
    for tier, threshold in sorted(RARITY_THRESHOLDS.items(),
                                   key=lambda x: x[1], reverse=True):
        if score >= threshold:
            return tier
    return RarityTier.COMMON


def compare_cards(
    card1: Dict[str, Any],
    card2: Dict[str, Any],
    category: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare two cards and determine the winner.

    If category is specified, compares that specific characteristic.
    Otherwise, compares overall power levels.

    Args:
        card1: First card
        card2: Second card
        category: Optional characteristic to compare

    Returns:
        Comparison result with winner information
    """
    result = {
        'card1': card1.get('image_name', 'Card 1'),
        'card2': card2.get('image_name', 'Card 2'),
        'category': category or 'power_level',
        'score1': 0,
        'score2': 0,
        'winner': None,
        'margin': 0,
        'details': {}
    }

    if category:
        # Compare specific characteristic
        char1 = card1.get('characteristics', {}).get(category, {})
        char2 = card2.get('characteristics', {}).get(category, {})

        score1 = char1.get('max_score', 0)
        score2 = char2.get('max_score', 0)

        result['score1'] = round(score1, 3)
        result['score2'] = round(score2, 3)
        result['details'] = {
            'prompt1': char1.get('max_prompt', 'N/A'),
            'prompt2': char2.get('max_prompt', 'N/A')
        }
    else:
        # Compare overall power levels
        score1 = calculate_power_level(card1)
        score2 = calculate_power_level(card2)

        result['score1'] = score1
        result['score2'] = score2

    # Determine winner
    if result['score1'] > result['score2']:
        result['winner'] = result['card1']
        result['margin'] = round(result['score1'] - result['score2'], 3)
    elif result['score2'] > result['score1']:
        result['winner'] = result['card2']
        result['margin'] = round(result['score2'] - result['score1'], 3)
    else:
        result['winner'] = 'TIE'
        result['margin'] = 0

    return result


def battle_cards(
    card1: Dict[str, Any],
    card2: Dict[str, Any],
    rounds: int = 5
) -> Dict[str, Any]:
    """
    Full battle between two cards across multiple characteristics.

    Each round compares a different characteristic.
    The card winning the most rounds wins the battle.

    Args:
        card1: First card
        card2: Second card
        rounds: Number of rounds (characteristics to compare)

    Returns:
        Battle results with round-by-round breakdown
    """
    # Get available characteristics
    chars1 = set(card1.get('characteristics', {}).keys())
    chars2 = set(card2.get('characteristics', {}).keys())
    common_chars = list(chars1 & chars2)

    if not common_chars:
        return {
            'error': 'No common characteristics to compare',
            'winner': None
        }

    # Select random characteristics for battle
    np.random.shuffle(common_chars)
    battle_chars = common_chars[:min(rounds, len(common_chars))]

    results = {
        'card1': card1.get('image_name', 'Card 1'),
        'card2': card2.get('image_name', 'Card 2'),
        'rounds': [],
        'wins1': 0,
        'wins2': 0,
        'ties': 0,
        'winner': None
    }

    for char in battle_chars:
        round_result = compare_cards(card1, card2, char)
        results['rounds'].append(round_result)

        if round_result['winner'] == results['card1']:
            results['wins1'] += 1
        elif round_result['winner'] == results['card2']:
            results['wins2'] += 1
        else:
            results['ties'] += 1

    # Determine overall winner
    if results['wins1'] > results['wins2']:
        results['winner'] = results['card1']
    elif results['wins2'] > results['wins1']:
        results['winner'] = results['card2']
    else:
        results['winner'] = 'TIE'

    return results


def calculate_collection_stats(cards: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics for a collection of cards.

    Args:
        cards: List of card dictionaries

    Returns:
        Collection statistics
    """
    if not cards:
        return {'error': 'No cards in collection'}

    # Calculate power levels for all cards
    power_levels = [calculate_power_level(card) for card in cards]

    # Calculate rarities
    rarities = [calculate_rarity_score(card, cards) for card in cards]
    rarity_scores = [r[0] for r in rarities]
    rarity_tiers = [r[1] for r in rarities]

    # Count tiers
    tier_counts = {}
    for tier in RarityTier:
        tier_counts[tier.value] = sum(1 for t in rarity_tiers if t == tier)

    # Find best cards
    sorted_by_power = sorted(
        zip(cards, power_levels),
        key=lambda x: x[1],
        reverse=True
    )

    sorted_by_rarity = sorted(
        zip(cards, rarity_scores),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        'total_cards': len(cards),
        'power_stats': {
            'mean': round(np.mean(power_levels), 2),
            'max': round(max(power_levels), 2),
            'min': round(min(power_levels), 2),
            'std': round(np.std(power_levels), 2)
        },
        'rarity_stats': {
            'mean': round(np.mean(rarity_scores), 2),
            'max': round(max(rarity_scores), 2),
            'min': round(min(rarity_scores), 2)
        },
        'tier_distribution': tier_counts,
        'top_power': [
            {'name': c.get('image_name'), 'power': p}
            for c, p in sorted_by_power[:5]
        ],
        'most_rare': [
            {'name': c.get('image_name'), 'rarity': r}
            for c, r in sorted_by_rarity[:5]
        ]
    }


def enhance_cards_with_gaming_stats(
    cards: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Add power level and rarity to all cards in a collection.

    Args:
        cards: List of card dictionaries

    Returns:
        Enhanced cards with gaming stats
    """
    enhanced_cards = []

    for card in cards:
        enhanced_card = card.copy()

        # Calculate power level
        power = calculate_power_level(card)
        enhanced_card['power_level'] = power

        # Calculate rarity
        rarity_score, rarity_tier = calculate_rarity_score(card, cards)
        enhanced_card['rarity_score'] = rarity_score
        enhanced_card['rarity_tier'] = rarity_tier.value
        enhanced_card['rarity_color'] = RARITY_COLORS[rarity_tier]
        enhanced_card['rarity_symbol'] = RARITY_SYMBOLS[rarity_tier]

        enhanced_cards.append(enhanced_card)

    return enhanced_cards


def print_battle_result(result: Dict[str, Any]) -> None:
    """Print a formatted battle result"""
    print("\n CARD BATTLE")
    print("=" * 60)
    print(f"\n  {result['card1']}  vs  {result['card2']}")
    print()

    for i, round_result in enumerate(result.get('rounds', []), 1):
        category = round_result['category'].replace('_', ' ').title()
        winner = round_result['winner']

        if winner == result['card1']:
            indicator = ""
        elif winner == result['card2']:
            indicator = ""
        else:
            indicator = ""

        print(f"  Round {i} ({category}):")
        print(f"    {result['card1']}: {round_result['score1']:.2f}")
        print(f"    {result['card2']}: {round_result['score2']:.2f}")
        print(f"    {indicator} Winner: {winner}")
        print()

    print("-" * 60)
    print(f"  FINAL SCORE: {result['wins1']} - {result['wins2']}")

    if result['winner'] == 'TIE':
        print("   IT'S A TIE!")
    else:
        print(f"   WINNER: {result['winner']}")
    print()


def print_comparison_result(result: Dict[str, Any]) -> None:
    """Print a formatted comparison result"""
    print("\n CARD COMPARISON")
    print("=" * 50)

    category = result['category'].replace('_', ' ').title()
    print(f"\n  Category: {category}")
    print()
    print(f"  {result['card1']}: {result['score1']}")
    print(f"  {result['card2']}: {result['score2']}")
    print()

    if result['winner'] == 'TIE':
        print("  Result: TIE!")
    else:
        print(f"  Winner: {result['winner']}")
        print(f"  Margin: +{result['margin']}")

    if result.get('details'):
        print()
        print("  Details:")
        if result['details'].get('prompt1'):
            print(f"    {result['card1']}: {result['details']['prompt1']}")
        if result['details'].get('prompt2'):
            print(f"    {result['card2']}: {result['details']['prompt2']}")
    print()


def print_collection_stats(stats: Dict[str, Any]) -> None:
    """Print formatted collection statistics"""
    print("\n COLLECTION STATISTICS")
    print("=" * 50)

    print(f"\n  Total Cards: {stats['total_cards']}")

    print("\n  Power Level Distribution:")
    ps = stats['power_stats']
    print(f"    Mean: {ps['mean']}")
    print(f"    Max:  {ps['max']}")
    print(f"    Min:  {ps['min']}")

    print("\n  Rarity Distribution:")
    for tier, count in stats['tier_distribution'].items():
        symbol = RARITY_SYMBOLS.get(RarityTier(tier), "")
        print(f"    {symbol} {tier.title()}: {count}")

    print("\n  Top Power Cards:")
    for card in stats['top_power']:
        print(f"    {card['name']}: {card['power']}")

    print("\n  Rarest Cards:")
    for card in stats['most_rare']:
        print(f"    {card['name']}: {card['rarity']:.1f}")
    print()
