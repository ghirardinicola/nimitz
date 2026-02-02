"""
Tests for NIMITZ Deck Module
"""

import pytest
import json
import tempfile
from pathlib import Path
from deck import (
    Deck,
    create_deck_from_cards,
    merge_decks,
    split_deck,
)


class TestDeckBasic:
    """Tests for basic Deck functionality"""

    def test_deck_init_empty(self):
        """Test creating empty deck"""
        deck = Deck(name="Test Deck")

        assert deck.name == "Test Deck"
        assert deck.size() == 0
        assert deck.is_empty()

    def test_deck_init_with_cards(self, sample_cards_collection):
        """Test creating deck with initial cards"""
        deck = Deck(name="Test", cards=sample_cards_collection)

        assert deck.size() == 3
        assert not deck.is_empty()

    def test_deck_init_copies_cards(self, sample_cards_collection):
        """Deck should copy cards, not reference original list"""
        original = sample_cards_collection.copy()
        deck = Deck(name="Test", cards=sample_cards_collection)

        # Modify deck
        deck.cards.append({'image_name': 'new.jpg'})

        # Original should be unchanged
        assert len(sample_cards_collection) == len(original)

    def test_deck_add_card(self, sample_card):
        """Test adding a card to deck"""
        deck = Deck(name="Test")

        result = deck.add_card(sample_card)

        assert result is True
        assert deck.size() == 1
        assert deck.get_card('test_photo.jpg') is not None

    def test_deck_add_duplicate_card(self, sample_card):
        """Adding duplicate card should return False"""
        deck = Deck(name="Test")
        deck.add_card(sample_card)

        result = deck.add_card(sample_card)

        assert result is False
        assert deck.size() == 1

    def test_deck_remove_card(self, sample_card):
        """Test removing a card from deck"""
        deck = Deck(name="Test")
        deck.add_card(sample_card)

        result = deck.remove_card('test_photo.jpg')

        assert result is True
        assert deck.size() == 0

    def test_deck_remove_nonexistent_card(self):
        """Removing nonexistent card should return False"""
        deck = Deck(name="Test")

        result = deck.remove_card('nonexistent.jpg')

        assert result is False

    def test_deck_get_card(self, sample_card, sample_card_2):
        """Test getting a card by name"""
        deck = Deck(name="Test")
        deck.add_card(sample_card)
        deck.add_card(sample_card_2)

        card = deck.get_card('test_photo.jpg')

        assert card is not None
        assert card['image_name'] == 'test_photo.jpg'

    def test_deck_get_card_not_found(self, sample_card):
        """Getting nonexistent card should return None"""
        deck = Deck(name="Test")
        deck.add_card(sample_card)

        card = deck.get_card('nonexistent.jpg')

        assert card is None

    def test_deck_clear(self, sample_cards_collection):
        """Test clearing all cards from deck"""
        deck = Deck(name="Test", cards=sample_cards_collection)

        deck.clear()

        assert deck.is_empty()
        assert deck.size() == 0


class TestDeckStats:
    """Tests for deck statistics"""

    def test_deck_get_stats(self, sample_cards_collection):
        """Test getting deck statistics"""
        deck = Deck(name="Test Deck", cards=sample_cards_collection)

        stats = deck.get_stats()

        assert stats['name'] == "Test Deck"
        assert stats['size'] == 3
        assert 'power' in stats
        assert 'rarity' in stats
        assert 'top_cards' in stats

    def test_deck_get_stats_empty(self):
        """Empty deck stats should have error"""
        deck = Deck(name="Empty")

        stats = deck.get_stats()

        assert stats['size'] == 0
        assert 'error' in stats

    def test_deck_stats_power_values(self, sample_cards_collection):
        """Power stats should have valid values"""
        deck = Deck(name="Test", cards=sample_cards_collection)

        stats = deck.get_stats()

        assert stats['power']['min'] <= stats['power']['mean'] <= stats['power']['max']
        assert stats['power']['total'] > 0


class TestDeckSerialization:
    """Tests for deck save/load functionality"""

    def test_deck_to_dict(self, sample_cards_collection):
        """Test converting deck to dictionary"""
        deck = Deck(name="Test", description="A test deck", cards=sample_cards_collection)

        data = deck.to_dict()

        assert data['name'] == "Test"
        assert data['description'] == "A test deck"
        assert len(data['cards']) == 3
        assert 'created_at' in data
        assert 'updated_at' in data

    def test_deck_from_dict(self):
        """Test creating deck from dictionary"""
        data = {
            'name': 'Imported',
            'description': 'From dict',
            'cards': [{'image_name': 'test.jpg'}],
            'created_at': '2024-01-01T00:00:00',
            'updated_at': '2024-01-01T00:00:00',
            'metadata': {'key': 'value'}
        }

        deck = Deck.from_dict(data)

        assert deck.name == 'Imported'
        assert deck.description == 'From dict'
        assert deck.size() == 1
        assert deck.metadata == {'key': 'value'}

    def test_deck_save_and_load(self, sample_cards_collection):
        """Test saving and loading deck to/from file"""
        deck = Deck(name="Saved Deck", cards=sample_cards_collection)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            # Save
            saved_path = deck.save(filepath)
            assert saved_path.exists()

            # Load
            loaded = Deck.load(filepath)

            assert loaded.name == deck.name
            assert loaded.size() == deck.size()
            assert loaded.cards[0]['image_name'] == deck.cards[0]['image_name']
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_deck_save_adds_json_extension(self, sample_card):
        """Save should add .json extension if missing"""
        deck = Deck(name="Test")
        deck.add_card(sample_card)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "deck_without_extension"
            saved_path = deck.save(str(filepath))

            assert saved_path.suffix == '.json'
            assert saved_path.exists()

    def test_deck_load_nonexistent_file(self):
        """Loading nonexistent file should raise FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            Deck.load('/nonexistent/path/deck.json')


class TestDeckSorting:
    """Tests for deck sorting functionality"""

    def test_deck_sort_by_power(self, sample_cards_collection):
        """Test sorting deck by power level"""
        deck = Deck(name="Test", cards=sample_cards_collection)

        deck.sort_by_power(descending=True)

        # First card should have highest power
        from gaming import calculate_power_level
        powers = [calculate_power_level(c) for c in deck.cards]
        assert powers == sorted(powers, reverse=True)

    def test_deck_sort_by_power_ascending(self, sample_cards_collection):
        """Test sorting deck by power level ascending"""
        deck = Deck(name="Test", cards=sample_cards_collection)

        deck.sort_by_power(descending=False)

        from gaming import calculate_power_level
        powers = [calculate_power_level(c) for c in deck.cards]
        assert powers == sorted(powers)

    def test_deck_filter_by_rarity(self, sample_cards_collection):
        """Test filtering cards by minimum rarity"""
        deck = Deck(name="Test", cards=sample_cards_collection)

        # Filter for at least uncommon
        filtered = deck.filter_by_rarity('uncommon')

        # All filtered cards should have rarity >= uncommon
        tier_order = ['common', 'uncommon', 'rare', 'epic', 'legendary']
        for card in filtered:
            tier_index = tier_order.index(card['rarity_tier'])
            assert tier_index >= tier_order.index('uncommon')


class TestDeckHelperFunctions:
    """Tests for deck helper functions"""

    def test_create_deck_from_cards(self, sample_cards_collection):
        """Test creating deck from cards list"""
        deck = create_deck_from_cards(
            sample_cards_collection,
            name="Created Deck",
            description="From helper"
        )

        assert deck.name == "Created Deck"
        assert deck.description == "From helper"
        assert deck.size() == 3

    def test_merge_decks(self, sample_card, sample_card_2):
        """Test merging two decks"""
        deck1 = Deck(name="Deck 1", cards=[sample_card])
        deck2 = Deck(name="Deck 2", cards=[sample_card_2])

        merged = merge_decks(deck1, deck2, name="Merged")

        assert merged.name == "Merged"
        assert merged.size() == 2

    def test_merge_decks_no_duplicates(self, sample_card):
        """Merging decks with same cards should not duplicate"""
        deck1 = Deck(name="Deck 1", cards=[sample_card])
        deck2 = Deck(name="Deck 2", cards=[sample_card])

        merged = merge_decks(deck1, deck2)

        assert merged.size() == 1

    def test_split_deck_random(self, sample_cards_collection):
        """Test splitting deck randomly"""
        deck = Deck(name="Original", cards=sample_cards_collection)

        splits = split_deck(deck, n=2, strategy="random")

        assert len(splits) == 2
        total_cards = sum(d.size() for d in splits)
        assert total_cards == deck.size()

    def test_split_deck_balanced(self, sample_cards_collection):
        """Test splitting deck with balanced strategy"""
        deck = Deck(name="Original", cards=sample_cards_collection)

        splits = split_deck(deck, n=2, strategy="balanced")

        assert len(splits) == 2
        # Each deck should have cards
        for split in splits:
            assert split.size() > 0

    def test_split_deck_empty(self):
        """Splitting empty deck should return empty list"""
        deck = Deck(name="Empty")

        splits = split_deck(deck, n=2)

        assert splits == []

    def test_split_deck_zero_parts(self, sample_cards_collection):
        """Splitting into 0 parts should return empty list"""
        deck = Deck(name="Test", cards=sample_cards_collection)

        splits = split_deck(deck, n=0)

        assert splits == []
