"""
Tests for NIMITZ Gaming Module
"""

import pytest
from gaming import (
    calculate_power_level,
    calculate_rarity_score,
    compare_cards,
    battle_cards,
    calculate_collection_stats,
    enhance_cards_with_gaming_stats,
    RarityTier,
    RARITY_THRESHOLDS,
    RARITY_COLORS,
)


class TestPowerLevel:
    """Tests for power level calculation"""

    def test_calculate_power_level_basic(self, sample_card):
        """Test basic power level calculation"""
        power = calculate_power_level(sample_card)

        assert isinstance(power, float)
        assert 0 <= power <= 100

    def test_calculate_power_level_strong_card(self, sample_card):
        """Strong card should have higher power"""
        power = calculate_power_level(sample_card)

        # Card has max=0.90, mean=0.69, high_conf=3/10
        # Expected: (0.90*0.4 + 0.69*0.3 + 0.3*0.3) * 100 = 65.7
        assert power > 60
        assert power < 75

    def test_calculate_power_level_weak_card(self, sample_card_weak):
        """Weak card should have lower power"""
        power = calculate_power_level(sample_card_weak)

        # Card has max=0.52, mean=0.46, high_conf=0/10
        assert power < 50

    def test_calculate_power_level_empty_card(self):
        """Empty card should return 0"""
        empty_card = {'feature_summary': {}}
        power = calculate_power_level(empty_card)

        assert power == 0

    def test_calculate_power_level_comparison(self, sample_card, sample_card_weak):
        """Strong card should beat weak card"""
        power_strong = calculate_power_level(sample_card)
        power_weak = calculate_power_level(sample_card_weak)

        assert power_strong > power_weak


class TestRarity:
    """Tests for rarity calculation"""

    def test_rarity_score_basic(self, sample_card, sample_cards_collection):
        """Test basic rarity score calculation"""
        score, tier = calculate_rarity_score(sample_card, sample_cards_collection)

        assert isinstance(score, float)
        assert 0 <= score <= 100
        assert isinstance(tier, RarityTier)

    def test_rarity_single_card(self, sample_card):
        """Single card collection should return default rarity"""
        score, tier = calculate_rarity_score(sample_card, [sample_card])

        assert score == 50.0
        assert tier == RarityTier.UNCOMMON

    def test_rarity_unique_card_is_rarer(self, sample_card, sample_card_2, sample_card_weak):
        """More unique cards should have higher rarity"""
        collection = [sample_card, sample_card_2, sample_card_weak]

        score_strong, _ = calculate_rarity_score(sample_card, collection)
        score_weak, _ = calculate_rarity_score(sample_card_weak, collection)

        # Different cards should have different rarity scores
        assert score_strong != score_weak

    def test_rarity_tier_enum(self):
        """Test RarityTier enum values"""
        assert RarityTier.COMMON.value == "common"
        assert RarityTier.UNCOMMON.value == "uncommon"
        assert RarityTier.RARE.value == "rare"
        assert RarityTier.EPIC.value == "epic"
        assert RarityTier.LEGENDARY.value == "legendary"

    def test_rarity_colors_defined(self):
        """All rarity tiers should have colors"""
        for tier in RarityTier:
            assert tier in RARITY_COLORS
            assert RARITY_COLORS[tier].startswith("#")


class TestCompareCards:
    """Tests for card comparison"""

    def test_compare_cards_power_level(self, sample_card, sample_card_weak):
        """Compare cards by overall power level"""
        result = compare_cards(sample_card, sample_card_weak)

        assert result['winner'] == sample_card['image_name']
        assert result['score1'] > result['score2']
        assert result['margin'] > 0
        assert result['category'] == 'power_level'

    def test_compare_cards_specific_category(self, sample_card, sample_card_2):
        """Compare cards by specific characteristic"""
        # sample_card has lighting max 0.85, sample_card_2 has 0.92
        result = compare_cards(sample_card, sample_card_2, category='lighting')

        assert result['category'] == 'lighting'
        assert result['winner'] == sample_card_2['image_name']
        assert 'details' in result

    def test_compare_cards_tie(self, sample_card):
        """Comparing card to itself should tie"""
        result = compare_cards(sample_card, sample_card)

        assert result['winner'] == 'TIE'
        assert result['margin'] == 0

    def test_compare_cards_result_structure(self, sample_card, sample_card_2):
        """Result should have all expected fields"""
        result = compare_cards(sample_card, sample_card_2)

        assert 'card1' in result
        assert 'card2' in result
        assert 'category' in result
        assert 'score1' in result
        assert 'score2' in result
        assert 'winner' in result
        assert 'margin' in result


class TestBattleCards:
    """Tests for card battles"""

    def test_battle_cards_basic(self, sample_card, sample_card_2):
        """Test basic battle between two cards"""
        result = battle_cards(sample_card, sample_card_2, rounds=3)

        assert 'rounds' in result
        assert len(result['rounds']) <= 3
        assert 'wins1' in result
        assert 'wins2' in result
        assert 'ties' in result
        assert 'winner' in result

    def test_battle_cards_total_rounds(self, sample_card, sample_card_2):
        """Wins + ties should equal total rounds"""
        result = battle_cards(sample_card, sample_card_2, rounds=5)

        total = result['wins1'] + result['wins2'] + result['ties']
        assert total == len(result['rounds'])

    def test_battle_cards_winner_has_more_wins(self, sample_card, sample_card_2):
        """Winner should have more wins (unless tie)"""
        result = battle_cards(sample_card, sample_card_2, rounds=5)

        if result['winner'] == result['card1']:
            assert result['wins1'] > result['wins2']
        elif result['winner'] == result['card2']:
            assert result['wins2'] > result['wins1']
        else:
            assert result['winner'] == 'TIE'
            assert result['wins1'] == result['wins2']

    def test_battle_cards_no_common_characteristics(self):
        """Battle with no common characteristics should return error"""
        card1 = {'image_name': 'a', 'characteristics': {'foo': {}}}
        card2 = {'image_name': 'b', 'characteristics': {'bar': {}}}

        result = battle_cards(card1, card2)

        assert 'error' in result
        assert result['winner'] is None


class TestCollectionStats:
    """Tests for collection statistics"""

    def test_collection_stats_basic(self, sample_cards_collection):
        """Test basic collection stats calculation"""
        stats = calculate_collection_stats(sample_cards_collection)

        assert stats['total_cards'] == 3
        assert 'power_stats' in stats
        assert 'rarity_stats' in stats
        assert 'tier_distribution' in stats
        assert 'top_power' in stats
        assert 'most_rare' in stats

    def test_collection_stats_empty(self):
        """Empty collection should return error"""
        stats = calculate_collection_stats([])

        assert 'error' in stats

    def test_collection_stats_power_values(self, sample_cards_collection):
        """Power stats should have valid values"""
        stats = calculate_collection_stats(sample_cards_collection)

        ps = stats['power_stats']
        assert ps['min'] <= ps['mean'] <= ps['max']
        assert ps['std'] >= 0

    def test_collection_stats_tier_distribution(self, sample_cards_collection):
        """Tier distribution should sum to total cards"""
        stats = calculate_collection_stats(sample_cards_collection)

        total_in_tiers = sum(stats['tier_distribution'].values())
        assert total_in_tiers == stats['total_cards']


class TestEnhanceCards:
    """Tests for enhancing cards with gaming stats"""

    def test_enhance_cards_adds_power_level(self, sample_cards_collection):
        """Enhanced cards should have power_level"""
        enhanced = enhance_cards_with_gaming_stats(sample_cards_collection)

        for card in enhanced:
            assert 'power_level' in card
            assert isinstance(card['power_level'], float)

    def test_enhance_cards_adds_rarity(self, sample_cards_collection):
        """Enhanced cards should have rarity fields"""
        enhanced = enhance_cards_with_gaming_stats(sample_cards_collection)

        for card in enhanced:
            assert 'rarity_score' in card
            assert 'rarity_tier' in card
            assert 'rarity_color' in card
            assert 'rarity_symbol' in card

    def test_enhance_cards_preserves_original(self, sample_cards_collection):
        """Original cards should not be modified"""
        original_names = [c['image_name'] for c in sample_cards_collection]

        enhanced = enhance_cards_with_gaming_stats(sample_cards_collection)

        # Original cards unchanged
        for card in sample_cards_collection:
            assert 'power_level' not in card

        # Enhanced cards have original data
        enhanced_names = [c['image_name'] for c in enhanced]
        assert enhanced_names == original_names

    def test_enhance_cards_empty_list(self):
        """Empty list should return empty list"""
        enhanced = enhance_cards_with_gaming_stats([])

        assert enhanced == []
