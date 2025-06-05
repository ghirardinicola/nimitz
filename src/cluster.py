"""
NIMITZ - Clustering Module
Image clustering and analysis functions
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Any

def cluster_images(
    combined_features: np.ndarray,
    method: str = 'kmeans',
    n_clusters: int = 5,
    **kwargs
) -> Tuple[np.ndarray, int]:
    """Cluster images using combined feature space"""
    print(f"🎲 NIMITZ: Attacking with {method.upper()}...")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(combined_features)
    
    # Apply clustering
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, **kwargs)
        cluster_labels = clusterer.fit_predict(features_scaled)
        final_n_clusters = n_clusters
        
    elif method == 'dbscan':
        eps = kwargs.get('eps', 0.5)
        min_samples = kwargs.get('min_samples', 2)
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = clusterer.fit_predict(features_scaled)
        final_n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    
    print(f"💥 NIMITZ: Attack completed - {final_n_clusters} zones conquered")
    return cluster_labels, final_n_clusters

def create_results_dataframe(
    image_paths: List[Path],
    cluster_labels: np.ndarray
) -> pd.DataFrame:
    """Create results DataFrame"""
    return pd.DataFrame({
        'image_path': [Path(p).name for p in image_paths],
        'cluster': cluster_labels,
        'full_path': [str(p) for p in image_paths]
    })

def analyze_cluster_characteristics(
    cluster_labels: np.ndarray,
    combined_features: np.ndarray,
    feature_names: List[str],
    characteristic_labels: Dict[str, List[str]],
    n_clusters: int,
    results_df: pd.DataFrame,
    top_k: int = 3
) -> Dict[int, Dict]:
    """Analyze dominant characteristics for each cluster"""
    print("\n📊 NIMITZ - INTELLIGENCE REPORT")
    print("=" * 60)
    
    cluster_analysis = {}
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_features = combined_features[cluster_mask]
        n_images = np.sum(cluster_mask)
        
        if n_images == 0:
            continue
            
        print(f"\n🎯 SECTOR {cluster_id} ({n_images} images under control)")
        
        # Calculate mean features for cluster
        mean_features = np.mean(cluster_features, axis=0)
        
        # Find top-k most important features
        top_indices = np.argsort(mean_features)[-top_k:][::-1]
        
        cluster_chars = []
        for i, feature_idx in enumerate(top_indices):
            feature_name = feature_names[feature_idx]
            score = mean_features[feature_idx]
            
            # Separate characteristic name and description
            char_name = feature_name.split('_')[0]
            description = ' '.join(feature_name.split('_')[1:])
            
            print(f"  ⚡ {i+1}. {char_name.upper()}: {description}")
            print(f"     🎯 Precision: {score:.3f}")
            
            cluster_chars.append({
                'characteristic': char_name,
                'description': description,
                'score': score
            })
        
        cluster_analysis[cluster_id] = {
            'n_images': n_images,
            'characteristics': cluster_chars
        }
        
        # Show sample images from cluster
        cluster_images = results_df[results_df['cluster'] == cluster_id]['image_path'].head(3)
        print(f"  📸 Samples: {', '.join(cluster_images.tolist())}")
        
    print(f"\n⚓ NIMITZ: Intelligence report completed")
    return cluster_analysis

def discover_characteristic_combinations(
    cluster_analysis: Dict[int, Dict],
    min_score_threshold: float = 0.3
) -> Dict[int, Dict]:
    """Discover interesting characteristic combinations in clusters"""
    print("\n🔍 NIMITZ - DISCOVERING SEMANTIC ROUTES")
    print("=" * 50)
    
    discovered_combinations = {}
    
    for cluster_id, analysis in cluster_analysis.items():
        chars = analysis['characteristics']
        
        # Filter significant characteristics
        significant_chars = [c for c in chars if c['score'] > min_score_threshold]
        
        if len(significant_chars) >= 2:
            # Create combination description
            combination_desc = []
            char_types = []
            
            for char in significant_chars:
                combination_desc.append(char['description'])
                char_types.append(char['characteristic'])
            
            combination_key = " + ".join(combination_desc)
            
            discovered_combinations[cluster_id] = {
                'description': combination_key,
                'characteristics': char_types,
                'n_images': analysis['n_images'],
                'scores': [c['score'] for c in significant_chars]
            }
            
            print(f"\n🌟 Sector {cluster_id}: {combination_key}")
            print(f"   📊 Images: {analysis['n_images']}")
            print(f"   🎯 Scores: {[f'{s:.3f}' for s in discovered_combinations[cluster_id]['scores']]}")
    
    print(f"\n⚓ NIMITZ: {len(discovered_combinations)} semantic routes discovered")
    return discovered_combinations

def generate_cluster_cards(
    cluster_analysis: Dict[int, Dict],
    results_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    combined_features: np.ndarray
) -> List[Dict]:
    """Generate trading card style data for each cluster"""
    print("🃏 NIMITZ: Generating collectible cards...")
    
    cards_data = []
    
    for cluster_id, analysis in cluster_analysis.items():
        # Calculate stats for the "card"
        cluster_images = results_df[results_df['cluster'] == cluster_id]
        
        # Top 3 characteristics as "stats"
        stats = {}
        for i, char in enumerate(analysis['characteristics'][:3]):
            stat_name = char['characteristic'].replace('_', ' ').title()
            stats[stat_name] = char['description']
        
        # Find representative image (closest to centroid)
        if len(cluster_images) > 0:
            cluster_mask = cluster_labels == cluster_id
            cluster_features = combined_features[cluster_mask]
            centroid = np.mean(cluster_features, axis=0)
            
            # Find image closest to centroid
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            representative_idx = np.argmin(distances)
            representative_image = cluster_images.iloc[representative_idx]['image_path']
        else:
            representative_image = "N/A"
        
        card_data = {
            'cluster_id': cluster_id,
            'name': f"Sector {cluster_id}",
            'type': 'Semantic Cluster',
            #'power_level': sum(stats.values()) // len(stats) if stats else 0,
            'population': analysis['n_images'],
            'representative_image': representative_image,
            'stats': stats,
            #'description': f"Cluster characterized by {', '.join([c['characteristic'] for c in analysis['characteristics'][:2]])}"
        }
        
        cards_data.append(card_data)
    
    print(f"🃏 NIMITZ: {len(cards_data)} cards generated")
    return cards_data