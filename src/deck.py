#!/usr/bin/env python3
"""
NIMITZ - Deck Management Module
Create, save, load, and manage custom card decks
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

from gaming import (
    calculate_power_level,
    calculate_rarity_score,
    enhance_cards_with_gaming_stats,
    RarityTier,
    RARITY_SYMBOLS
)


class Deck:
    """
    A collection of cards that can be saved, loaded, and used for battles.
    """

    def __init__(
        self,
        name: str = "My Deck",
        description: str = "",
        cards: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize a new deck.

        Args:
            name: Deck name
            description: Optional deck description
            cards: Optional initial list of cards
        """
        self.name = name
        self.description = description
        self.cards: List[Dict[str, Any]] = cards if cards else []
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.metadata: Dict[str, Any] = {}

    def add_card(self, card: Dict[str, Any]) -> bool:
        """
        Add a card to the deck.

        Args:
            card: Card dictionary to add

        Returns:
            True if card was added, False if already exists
        """
        # Check if card already exists by image name
        card_name = card.get('image_name')
        for existing in self.cards:
            if existing.get('image_name') == card_name:
                return False

        self.cards.append(card)
        self.updated_at = datetime.now().isoformat()
        return True

    def remove_card(self, image_name: str) -> bool:
        """
        Remove a card from the deck by image name.

        Args:
            image_name: Name of the image to remove

        Returns:
            True if card was removed, False if not found
        """
        for i, card in enumerate(self.cards):
            if card.get('image_name') == image_name:
                self.cards.pop(i)
                self.updated_at = datetime.now().isoformat()
                return True
        return False

    def get_card(self, image_name: str) -> Optional[Dict[str, Any]]:
        """Get a card by image name"""
        for card in self.cards:
            if card.get('image_name') == image_name:
                return card
        return None

    def get_random_card(self) -> Optional[Dict[str, Any]]:
        """Get a random card from the deck"""
        if not self.cards:
            return None
        return np.random.choice(self.cards)

    def get_random_cards(self, n: int) -> List[Dict[str, Any]]:
        """Get n random cards from the deck (without replacement)"""
        if not self.cards:
            return []
        n = min(n, len(self.cards))
        indices = np.random.choice(len(self.cards), n, replace=False)
        return [self.cards[i] for i in indices]

    def size(self) -> int:
        """Return number of cards in deck"""
        return len(self.cards)

    def is_empty(self) -> bool:
        """Check if deck is empty"""
        return len(self.cards) == 0

    def clear(self) -> None:
        """Remove all cards from deck"""
        self.cards = []
        self.updated_at = datetime.now().isoformat()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get deck statistics including power levels and rarity.

        Returns:
            Dictionary with deck statistics
        """
        if not self.cards:
            return {
                'name': self.name,
                'size': 0,
                'error': 'Deck is empty'
            }

        # Enhance cards with gaming stats
        enhanced = enhance_cards_with_gaming_stats(self.cards)

        power_levels = [c.get('power_level', 0) for c in enhanced]
        rarity_scores = [c.get('rarity_score', 0) for c in enhanced]

        # Count tiers
        tier_counts = {tier.value: 0 for tier in RarityTier}
        for card in enhanced:
            tier = card.get('rarity_tier', 'common')
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

        # Find strongest cards
        sorted_cards = sorted(
            enhanced,
            key=lambda c: c.get('power_level', 0),
            reverse=True
        )

        return {
            'name': self.name,
            'description': self.description,
            'size': len(self.cards),
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'power': {
                'total': round(sum(power_levels), 2),
                'mean': round(np.mean(power_levels), 2),
                'max': round(max(power_levels), 2),
                'min': round(min(power_levels), 2)
            },
            'rarity': {
                'mean': round(np.mean(rarity_scores), 2),
                'distribution': tier_counts
            },
            'top_cards': [
                {'name': c.get('image_name'), 'power': c.get('power_level')}
                for c in sorted_cards[:5]
            ]
        }

    def sort_by_power(self, descending: bool = True) -> None:
        """Sort cards by power level"""
        enhanced = enhance_cards_with_gaming_stats(self.cards)
        power_map = {
            c.get('image_name'): c.get('power_level', 0)
            for c in enhanced
        }
        self.cards.sort(
            key=lambda c: power_map.get(c.get('image_name'), 0),
            reverse=descending
        )

    def sort_by_rarity(self, descending: bool = True) -> None:
        """Sort cards by rarity score"""
        enhanced = enhance_cards_with_gaming_stats(self.cards)
        rarity_map = {
            c.get('image_name'): c.get('rarity_score', 0)
            for c in enhanced
        }
        self.cards.sort(
            key=lambda c: rarity_map.get(c.get('image_name'), 0),
            reverse=descending
        )

    def filter_by_rarity(self, min_tier: str) -> List[Dict[str, Any]]:
        """
        Filter cards by minimum rarity tier.

        Args:
            min_tier: Minimum rarity tier ('common', 'uncommon', 'rare', 'epic', 'legendary')

        Returns:
            List of cards meeting the rarity requirement
        """
        tier_order = ['common', 'uncommon', 'rare', 'epic', 'legendary']
        min_index = tier_order.index(min_tier.lower()) if min_tier.lower() in tier_order else 0

        enhanced = enhance_cards_with_gaming_stats(self.cards)
        return [
            c for c in enhanced
            if tier_order.index(c.get('rarity_tier', 'common')) >= min_index
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert deck to dictionary for serialization"""
        return {
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'metadata': self.metadata,
            'cards': self.cards
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Deck':
        """Create deck from dictionary"""
        deck = cls(
            name=data.get('name', 'Imported Deck'),
            description=data.get('description', ''),
            cards=data.get('cards', [])
        )
        deck.created_at = data.get('created_at', deck.created_at)
        deck.updated_at = data.get('updated_at', deck.updated_at)
        deck.metadata = data.get('metadata', {})
        return deck

    def save(self, filepath: str) -> Path:
        """
        Save deck to JSON file.

        Args:
            filepath: Path to save file

        Returns:
            Path to saved file
        """
        path = Path(filepath)
        if not path.suffix:
            path = path.with_suffix('.json')

        self.updated_at = datetime.now().isoformat()

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)

        return path

    @classmethod
    def load(cls, filepath: str) -> 'Deck':
        """
        Load deck from JSON file.

        Args:
            filepath: Path to deck file

        Returns:
            Loaded Deck instance
        """
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Deck file not found: {filepath}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return cls.from_dict(data)


def create_deck_from_cards(
    cards: List[Dict[str, Any]],
    name: str = "New Deck",
    description: str = ""
) -> Deck:
    """
    Create a new deck from a list of cards.

    Args:
        cards: List of card dictionaries
        name: Deck name
        description: Deck description

    Returns:
        New Deck instance
    """
    return Deck(name=name, description=description, cards=cards.copy())


def merge_decks(
    deck1: Deck,
    deck2: Deck,
    name: str = "Merged Deck"
) -> Deck:
    """
    Merge two decks into a new deck.

    Duplicate cards (by image name) are included only once.

    Args:
        deck1: First deck
        deck2: Second deck
        name: Name for merged deck

    Returns:
        New merged Deck
    """
    merged = Deck(name=name, description=f"Merged from {deck1.name} and {deck2.name}")

    # Add all cards from deck1
    for card in deck1.cards:
        merged.add_card(card)

    # Add non-duplicate cards from deck2
    for card in deck2.cards:
        merged.add_card(card)

    return merged


def split_deck(
    deck: Deck,
    n: int,
    strategy: str = "random"
) -> List[Deck]:
    """
    Split a deck into n smaller decks.

    Args:
        deck: Deck to split
        n: Number of decks to create
        strategy: Split strategy ('random', 'sequential', 'balanced')

    Returns:
        List of new Deck instances
    """
    if n <= 0 or deck.is_empty():
        return []

    cards = deck.cards.copy()

    if strategy == "random":
        np.random.shuffle(cards)
    elif strategy == "balanced":
        # Sort by power and distribute evenly
        enhanced = enhance_cards_with_gaming_stats(cards)
        cards = sorted(enhanced, key=lambda c: c.get('power_level', 0), reverse=True)

    # Distribute cards
    decks = [Deck(name=f"{deck.name} Part {i+1}") for i in range(n)]

    for i, card in enumerate(cards):
        decks[i % n].add_card(card)

    return decks


def print_deck_info(deck: Deck) -> None:
    """Print formatted deck information"""
    stats = deck.get_stats()

    print(f"\n DECK: {stats['name']}")
    print("=" * 50)

    if stats.get('description'):
        print(f"  Description: {stats['description']}")

    print(f"\n  Cards: {stats['size']}")
    print(f"  Created: {stats['created_at'][:10]}")

    if 'power' in stats:
        print(f"\n  Power Stats:")
        print(f"    Total: {stats['power']['total']}")
        print(f"    Mean:  {stats['power']['mean']}")
        print(f"    Best:  {stats['power']['max']}")

    if 'rarity' in stats:
        print(f"\n  Rarity Distribution:")
        for tier, count in stats['rarity']['distribution'].items():
            if count > 0:
                symbol = RARITY_SYMBOLS.get(RarityTier(tier), "")
                print(f"    {symbol} {tier.title()}: {count}")

    if stats.get('top_cards'):
        print(f"\n  Top Cards:")
        for card in stats['top_cards']:
            print(f"    {card['name']}: {card['power']}")

    print()


def list_cards_in_deck(deck: Deck, show_stats: bool = True) -> None:
    """Print list of all cards in deck"""
    print(f"\n CARDS IN {deck.name.upper()}")
    print("=" * 60)

    if deck.is_empty():
        print("  (empty deck)")
        return

    enhanced = enhance_cards_with_gaming_stats(deck.cards)

    for i, card in enumerate(enhanced, 1):
        name = card.get('image_name', 'Unknown')
        power = card.get('power_level', 0)
        tier = card.get('rarity_tier', 'common')
        symbol = card.get('rarity_symbol', '')

        if show_stats:
            print(f"  {i:3}. {symbol} {name}")
            print(f"       Power: {power:.1f} | Rarity: {tier.title()}")
        else:
            print(f"  {i:3}. {name}")

    print(f"\n  Total: {deck.size()} cards")
    print()
