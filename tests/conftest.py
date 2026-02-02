"""
NIMITZ Test Fixtures
Common fixtures for testing gaming, deck, and export modules
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_card():
    """A single sample card with all required fields"""
    return {
        'image_name': 'test_photo.jpg',
        'image_path': '/tmp/test_photo.jpg',
        'image_index': 0,
        'characteristics': {
            'lighting': {
                'scores': [0.85, 0.62, 0.74, 0.55],
                'prompts': [
                    'natural daylight',
                    'golden hour warm light',
                    'studio flash lighting',
                    'dramatic side lighting'
                ],
                'max_score': 0.85,
                'max_prompt': 'natural daylight',
                'max_prompt_index': 0,
                'mean_score': 0.69,
                'std_score': 0.11
            },
            'composition': {
                'scores': [0.72, 0.68, 0.81],
                'prompts': [
                    'rule of thirds composition',
                    'centered symmetrical composition',
                    'leading lines composition'
                ],
                'max_score': 0.81,
                'max_prompt': 'leading lines composition',
                'max_prompt_index': 2,
                'mean_score': 0.74,
                'std_score': 0.05
            },
            'mood': {
                'scores': [0.90, 0.45, 0.60],
                'prompts': [
                    'bright and cheerful mood',
                    'dark and moody atmosphere',
                    'calm and peaceful'
                ],
                'max_score': 0.90,
                'max_prompt': 'bright and cheerful mood',
                'max_prompt_index': 0,
                'mean_score': 0.65,
                'std_score': 0.19
            }
        },
        'feature_summary': {
            'total_features': 10,
            'overall_max': 0.90,
            'overall_mean': 0.69,
            'overall_std': 0.12,
            'high_confidence_features': 3,
            'medium_confidence_features': 5,
            'low_confidence_features': 2
        },
        'dominant_features': [
            {
                'characteristic': 'mood',
                'prompt': 'bright and cheerful mood',
                'score': 0.90,
                'confidence': 'high'
            },
            {
                'characteristic': 'lighting',
                'prompt': 'natural daylight',
                'score': 0.85,
                'confidence': 'high'
            },
            {
                'characteristic': 'composition',
                'prompt': 'leading lines composition',
                'score': 0.81,
                'confidence': 'high'
            }
        ]
    }


@pytest.fixture
def sample_card_2():
    """A second sample card for comparison tests"""
    return {
        'image_name': 'test_photo_2.jpg',
        'image_path': '/tmp/test_photo_2.jpg',
        'image_index': 1,
        'characteristics': {
            'lighting': {
                'scores': [0.55, 0.92, 0.64, 0.45],
                'prompts': [
                    'natural daylight',
                    'golden hour warm light',
                    'studio flash lighting',
                    'dramatic side lighting'
                ],
                'max_score': 0.92,
                'max_prompt': 'golden hour warm light',
                'max_prompt_index': 1,
                'mean_score': 0.64,
                'std_score': 0.17
            },
            'composition': {
                'scores': [0.65, 0.88, 0.71],
                'prompts': [
                    'rule of thirds composition',
                    'centered symmetrical composition',
                    'leading lines composition'
                ],
                'max_score': 0.88,
                'max_prompt': 'centered symmetrical composition',
                'max_prompt_index': 1,
                'mean_score': 0.75,
                'std_score': 0.10
            },
            'mood': {
                'scores': [0.40, 0.85, 0.50],
                'prompts': [
                    'bright and cheerful mood',
                    'dark and moody atmosphere',
                    'calm and peaceful'
                ],
                'max_score': 0.85,
                'max_prompt': 'dark and moody atmosphere',
                'max_prompt_index': 1,
                'mean_score': 0.58,
                'std_score': 0.19
            }
        },
        'feature_summary': {
            'total_features': 10,
            'overall_max': 0.92,
            'overall_mean': 0.66,
            'overall_std': 0.15,
            'high_confidence_features': 2,
            'medium_confidence_features': 6,
            'low_confidence_features': 2
        },
        'dominant_features': [
            {
                'characteristic': 'lighting',
                'prompt': 'golden hour warm light',
                'score': 0.92,
                'confidence': 'high'
            },
            {
                'characteristic': 'composition',
                'prompt': 'centered symmetrical composition',
                'score': 0.88,
                'confidence': 'high'
            },
            {
                'characteristic': 'mood',
                'prompt': 'dark and moody atmosphere',
                'score': 0.85,
                'confidence': 'high'
            }
        ]
    }


@pytest.fixture
def sample_card_weak():
    """A weaker card for comparison tests"""
    return {
        'image_name': 'weak_photo.jpg',
        'image_path': '/tmp/weak_photo.jpg',
        'image_index': 2,
        'characteristics': {
            'lighting': {
                'scores': [0.45, 0.42, 0.44, 0.35],
                'prompts': [
                    'natural daylight',
                    'golden hour warm light',
                    'studio flash lighting',
                    'dramatic side lighting'
                ],
                'max_score': 0.45,
                'max_prompt': 'natural daylight',
                'max_prompt_index': 0,
                'mean_score': 0.42,
                'std_score': 0.04
            },
            'composition': {
                'scores': [0.52, 0.48, 0.51],
                'prompts': [
                    'rule of thirds composition',
                    'centered symmetrical composition',
                    'leading lines composition'
                ],
                'max_score': 0.52,
                'max_prompt': 'rule of thirds composition',
                'max_prompt_index': 0,
                'mean_score': 0.50,
                'std_score': 0.02
            },
            'mood': {
                'scores': [0.50, 0.45, 0.40],
                'prompts': [
                    'bright and cheerful mood',
                    'dark and moody atmosphere',
                    'calm and peaceful'
                ],
                'max_score': 0.50,
                'max_prompt': 'bright and cheerful mood',
                'max_prompt_index': 0,
                'mean_score': 0.45,
                'std_score': 0.04
            }
        },
        'feature_summary': {
            'total_features': 10,
            'overall_max': 0.52,
            'overall_mean': 0.46,
            'overall_std': 0.05,
            'high_confidence_features': 0,
            'medium_confidence_features': 4,
            'low_confidence_features': 6
        },
        'dominant_features': [
            {
                'characteristic': 'composition',
                'prompt': 'rule of thirds composition',
                'score': 0.52,
                'confidence': 'medium'
            }
        ]
    }


@pytest.fixture
def sample_cards_collection(sample_card, sample_card_2, sample_card_weak):
    """Collection of multiple cards for testing"""
    return [sample_card, sample_card_2, sample_card_weak]
