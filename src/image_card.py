"""
NIMITZ - Image Cards Generator
Generate detailed cards for every image with all feature values
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

def generate_image_cards_data(
    image_paths: List[Path],
    similarity_matrices: Dict[str, np.ndarray],
    characteristic_labels: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    """
    Generate detailed card data for every image with all feature values
    
    Args:
        image_paths: List of image file paths
        similarity_matrices: Dict of similarity matrices per characteristic
        characteristic_labels: Dict of characteristic labels
    
    Returns:
        List of image card dictionaries
    """
    print("🃏 NIMITZ: Generating detailed image cards...")
    
    cards_data = []
    
    for img_idx, img_path in enumerate(image_paths):
        img_name = Path(img_path).name
        
        # Create card for this image
        card = {
            'image_name': img_name,
            'image_path': str(img_path),
            'image_index': img_idx,
            'characteristics': {},
            'feature_summary': {},
            'dominant_features': []
        }
        
        # Extract all feature values for this image
        all_feature_scores = []
        
        for char_name, similarity_matrix in similarity_matrices.items():
            char_similarities = similarity_matrix[img_idx]  # Get this image's similarities
            prompts = characteristic_labels[char_name]
            
            # Store detailed characteristic info
            char_info = {
                'scores': char_similarities.tolist(),
                'prompts': prompts,
                'max_score': float(np.max(char_similarities)),
                'max_prompt_index': int(np.argmax(char_similarities)),
                'max_prompt': prompts[np.argmax(char_similarities)],
                'mean_score': float(np.mean(char_similarities)),
                'std_score': float(np.std(char_similarities))
            }
            
            card['characteristics'][char_name] = char_info
            
            # Collect for overall analysis
            all_feature_scores.extend(char_similarities.tolist())
        
        # Overall feature summary
        card['feature_summary'] = {
            'total_features': len(all_feature_scores),
            'overall_max': float(np.max(all_feature_scores)),
            'overall_mean': float(np.mean(all_feature_scores)),
            'overall_std': float(np.std(all_feature_scores)),
            'high_confidence_features': sum(1 for score in all_feature_scores if score > 0.7),
            'medium_confidence_features': sum(1 for score in all_feature_scores if 0.4 <= score <= 0.7),
            'low_confidence_features': sum(1 for score in all_feature_scores if score < 0.4)
        }
        
        # Find dominant features across all characteristics
        dominant_features = []
        for char_name, char_info in card['characteristics'].items():
            if char_info['max_score'] > 0.5:  # Threshold for "dominant"
                dominant_features.append({
                    'characteristic': char_name,
                    'prompt': char_info['max_prompt'],
                    'score': char_info['max_score'],
                    'confidence': 'high' if char_info['max_score'] > 0.7 else 'medium'
                })
        
        # Sort by score
        dominant_features.sort(key=lambda x: x['score'], reverse=True)
        card['dominant_features'] = dominant_features[:5]  # Top 5
        
        cards_data.append(card)
    
    print(f"🃏 NIMITZ: Generated {len(cards_data)} image cards")
    return cards_data

def print_simple_image_cards(
    cards_data: List[Dict[str, Any]], 
    max_cards: Optional[int] = None,
    show_all_scores: bool = False
) -> None:
    """
    Simple direct printing of image cards - guaranteed to work
    
    Args:
        cards_data: List of image card data
        max_cards: Maximum number of cards to print (None for all)
        show_all_scores: Whether to show all individual scores
    """
    print("🃏 NIMITZ - IMAGE CARD COLLECTION")
    print("=" * 80)
    
    cards_to_print = cards_data[:max_cards] if max_cards else cards_data
    
    for i, card in enumerate(cards_to_print, 1):
        print(f"\n🎯 CARD #{i}: {card['image_name']}")
        print("─" * 60)
        
        # Basic info
        print(f"📁 Path: {card['image_path']}")
        
        # Feature summary
        summary = card['feature_summary']
        print(f"📊 Feature Summary:")
        print(f"   • Total Features: {summary['total_features']}")
        print(f"   • Overall Max Score: {summary['overall_max']:.3f}")
        print(f"   • Overall Mean: {summary['overall_mean']:.3f}")
        print(f"   • High Confidence: {summary['high_confidence_features']} features")
        print(f"   • Medium Confidence: {summary['medium_confidence_features']} features")
        print(f"   • Low Confidence: {summary['low_confidence_features']} features")
        
        # Dominant features
        if card['dominant_features']:
            print(f"\n⚡ DOMINANT FEATURES:")
            for j, feature in enumerate(card['dominant_features'], 1):
                confidence_icon = "🔥" if feature['confidence'] == 'high' else "⭐"
                print(f"   {j}. {confidence_icon} {feature['characteristic'].upper()}: {feature['prompt']}")
                print(f"      Score: {feature['score']:.3f} ({feature['confidence']} confidence)")
        
        # Detailed characteristic breakdown
        print(f"\n📋 ALL CHARACTERISTIC SCORES:")
        for char_name, char_info in card['characteristics'].items():
            print(f"\n   🎯 {char_name.upper().replace('_', ' ')}")
            print(f"      Best Match: {char_info['max_score']:.3f} - {char_info['max_prompt']}")
            print(f"      Average: {char_info['mean_score']:.3f} ± {char_info['std_score']:.3f}")
            
            if show_all_scores:
                print(f"      All scores:")
                for k, (prompt, score) in enumerate(zip(char_info['prompts'], char_info['scores'])):
                    print(f"        {k+1}. {score:.3f} - {prompt}")
        
        if i < len(cards_to_print):
            print("\n" + "=" * 80)
    
    print(f"\n⚓ NIMITZ: Printed {len(cards_to_print)} image cards")

def print_all_feature_values(
    image_paths: List[Path],
    similarity_matrices: Dict[str, np.ndarray],
    characteristic_labels: Dict[str, List[str]],
    max_images: Optional[int] = None,
    min_score_highlight: float = 0.7
) -> None:
    """
    Print all feature values for all images in a simple table format
    
    Args:
        image_paths: List of image paths
        similarity_matrices: Similarity matrices for each characteristic
        characteristic_labels: Labels for each characteristic
        max_images: Maximum number of images to print
        min_score_highlight: Minimum score to highlight with special marking
    """
    print("📊 NIMITZ - ALL FEATURE VALUES")
    print("=" * 100)
    
    images_to_process = image_paths[:max_images] if max_images else image_paths
    
    for img_idx, img_path in enumerate(images_to_process):
        img_name = Path(img_path).name
        
        print(f"\n🖼️  IMAGE: {img_name}")
        print("-" * 80)
        
        # Process each characteristic
        for char_name, similarity_matrix in similarity_matrices.items():
            char_similarities = similarity_matrix[img_idx]
            prompts = characteristic_labels[char_name]
            
            print(f"\n🎯 {char_name.upper().replace('_', ' ')}")
            
            # Sort by score for better readability
            scored_prompts = list(zip(prompts, char_similarities))
            scored_prompts.sort(key=lambda x: x[1], reverse=True)
            
            for prompt, score in scored_prompts:
                # Highlight high scores
                if score >= min_score_highlight:
                    print(f"   🔥 {score:.3f} - {prompt}")
                elif score >= 0.5:
                    print(f"   ⭐ {score:.3f} - {prompt}")
                else:
                    print(f"      {score:.3f} - {prompt}")
        
        print("=" * 100)

def create_visual_image_cards(
    cards_data: List[Dict[str, Any]],
    output_dir: str = "image_cards",
    cards_per_page: int = 6
) -> None:
    """
    Create visual image cards as PNG files
    
    Args:
        cards_data: List of image card data
        output_dir: Directory to save card images
        cards_per_page: Number of cards per page/image
    """
    print("🎨 NIMITZ: Creating visual image cards...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Calculate number of pages needed
    n_pages = (len(cards_data) + cards_per_page - 1) // cards_per_page
    
    for page in range(n_pages):
        start_idx = page * cards_per_page
        end_idx = min(start_idx + cards_per_page, len(cards_data))
        page_cards = cards_data[start_idx:end_idx]
        
        # Create figure
        fig = plt.figure(figsize=(20, 16))
        
        # Calculate grid layout
        cols = 3 if cards_per_page >= 6 else 2
        rows = (len(page_cards) + cols - 1) // cols
        
        for i, card in enumerate(page_cards):
            ax = plt.subplot(rows, cols, i + 1)
            
            # Create card background
            rect = patches.Rectangle((0, 0), 1, 1, linewidth=2, 
                                   edgecolor='navy', facecolor='lightblue', alpha=0.3)
            ax.add_patch(rect)
            
            # Card title
            ax.text(0.5, 0.95, card['image_name'], 
                   ha='center', va='top', fontsize=12, fontweight='bold',
                   transform=ax.transAxes)
            
            # Feature summary
            summary = card['feature_summary']
            ax.text(0.05, 0.85, f"Max Score: {summary['overall_max']:.3f}", 
                   ha='left', va='top', fontsize=9, transform=ax.transAxes)
            ax.text(0.05, 0.80, f"Mean: {summary['overall_mean']:.3f}", 
                   ha='left', va='top', fontsize=9, transform=ax.transAxes)
            ax.text(0.05, 0.75, f"High Conf: {summary['high_confidence_features']}", 
                   ha='left', va='top', fontsize=9, transform=ax.transAxes)
            
            # Dominant features
            y_pos = 0.67
            ax.text(0.05, y_pos, "Dominant Features:", 
                   ha='left', va='top', fontsize=10, fontweight='bold',
                   transform=ax.transAxes)
            
            features_to_show = card['dominant_features'][:4]  # Top 4 for space
            for j, feature in enumerate(features_to_show):
                y_pos -= 0.08
                color = 'red' if feature['confidence'] == 'high' else 'orange'
                ax.text(0.05, y_pos, 
                       f"• {feature['characteristic']}: {feature['score']:.3f}", 
                       ha='left', va='top', fontsize=8, color=color,
                       transform=ax.transAxes)
            
            # Remove axes
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            
        plt.tight_layout()
        
        # Save page
        page_filename = output_path / f"image_cards_page_{page+1}.png"
        plt.savefig(page_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📄 Page {page+1}/{n_pages} saved: {page_filename}")
    
    print(f"🎨 NIMITZ: Visual cards completed in {output_path}")

def export_cards_to_csv(
    cards_data: List[Dict[str, Any]], 
    output_file: str = "image_cards.csv"
) -> pd.DataFrame:
    """
    Export image cards data to CSV format
    
    Args:
        cards_data: List of image card data
        output_file: Output CSV filename
    
    Returns:
        DataFrame with card data
    """
    print("📊 NIMITZ: Exporting cards to CSV...")
    
    # Flatten card data for CSV
    flattened_data = []
    
    for card in cards_data:
        row = {
            'image_name': card['image_name'],
            'image_path': card['image_path'],
            'overall_max_score': card['feature_summary']['overall_max'],
            'overall_mean_score': card['feature_summary']['overall_mean'],
            'high_confidence_features': card['feature_summary']['high_confidence_features'],
            'medium_confidence_features': card['feature_summary']['medium_confidence_features'],
            'low_confidence_features': card['feature_summary']['low_confidence_features']
        }
        
        # Add dominant features as columns
        for i, feature in enumerate(card['dominant_features'][:5], 1):
            row[f'dominant_{i}_characteristic'] = feature['characteristic']
            row[f'dominant_{i}_prompt'] = feature['prompt']
            row[f'dominant_{i}_score'] = feature['score']
            row[f'dominant_{i}_confidence'] = feature['confidence']
        
        # Add detailed characteristic scores
        for char_name, char_info in card['characteristics'].items():
            row[f'{char_name}_max_score'] = char_info['max_score']
            row[f'{char_name}_max_prompt'] = char_info['max_prompt']
            row[f'{char_name}_mean_score'] = char_info['mean_score']
            row[f'{char_name}_std_score'] = char_info['std_score']
        
        flattened_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(flattened_data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"📊 NIMITZ: Cards exported to {output_file}")
    
    return df

def find_similar_images(
    cards_data: List[Dict[str, Any]], 
    target_image_name: str,
    similarity_threshold: float = 0.8,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Find images similar to a target image based on feature patterns
    
    Args:
        cards_data: List of image card data
        target_image_name: Name of target image
        similarity_threshold: Minimum similarity threshold
        top_k: Number of similar images to return
    
    Returns:
        List of similar image cards with similarity scores
    """
    print(f"🔍 NIMITZ: Finding images similar to {target_image_name}...")
    
    # Find target image
    target_card = None
    for card in cards_data:
        if card['image_name'] == target_image_name:
            target_card = card
            break
    
    if target_card is None:
        print(f"❌ Target image '{target_image_name}' not found")
        return []
    
    # Extract target feature vector
    target_features = []
    for char_name in sorted(target_card['characteristics'].keys()):
        char_info = target_card['characteristics'][char_name]
        target_features.extend(char_info['scores'])
    
    target_vector = np.array(target_features)
    
    # Calculate similarities
    similarities = []
    
    for card in cards_data:
        if card['image_name'] == target_image_name:
            continue  # Skip self
        
        # Extract feature vector
        card_features = []
        for char_name in sorted(card['characteristics'].keys()):
            char_info = card['characteristics'][char_name]
            card_features.extend(char_info['scores'])
        
        card_vector = np.array(card_features)
        
        # Calculate cosine similarity
        similarity = np.dot(target_vector, card_vector) / (
            np.linalg.norm(target_vector) * np.linalg.norm(card_vector)
        )
        
        similarities.append({
            'card': card,
            'similarity': similarity
        })
    
    # Sort by similarity and filter
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Filter and return top-k
    similar_images = []
    for sim_data in similarities:
        if len(similar_images) >= top_k:
            break
        if sim_data['similarity'] >= similarity_threshold:
            similar_images.append({
                'image_card': sim_data['card'],
                'similarity_score': sim_data['similarity']
            })
    
    print(f"🎯 Found {len(similar_images)} similar images")
    return similar_images