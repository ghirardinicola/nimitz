"""
NIMITZ - Visualization Module
Plotting and visualization functions for cluster analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from typing import Dict, List, Optional

def visualize_clusters(
    combined_features: np.ndarray,
    cluster_labels: np.ndarray,
    n_clusters: int,
    cluster_analysis: Optional[Dict] = None,
    characteristics: Optional[Dict[str, List[str]]] = None,
    save_path: Optional[str] = None,
    show_feature_importance: bool = True
) -> None:
    """Visualize clusters and feature importance"""
    print("📈 NIMITZ: Generating tactical map...")
    
    # Create subplots
    if show_feature_importance and cluster_analysis and characteristics:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot 1: Clusters in PCA space
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(combined_features)
    
    scatter = ax1.scatter(features_2d[:, 0], features_2d[:, 1], 
                        c=cluster_labels, cmap='tab10', alpha=0.7, s=50)
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax1.set_title('🚢 NIMITZ - Semantic Map (PCA 2D)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Sector')
    
    # Plot 2: Feature importance per cluster (if requested)
    if show_feature_importance and cluster_analysis and characteristics:
        # Create importance matrix
        importance_matrix = np.zeros((n_clusters, len(characteristics)))
        char_names = list(characteristics.keys())
        
        for cluster_id, analysis in cluster_analysis.items():
            if cluster_id < n_clusters:  # Safety check
                for char_info in analysis['characteristics']:
                    if char_info['characteristic'] in char_names:
                        char_idx = char_names.index(char_info['characteristic'])
                        importance_matrix[cluster_id, char_idx] = char_info['score']
        
        # Heatmap
        sns.heatmap(importance_matrix, 
                   xticklabels=[c.replace('_', ' ').title() for c in char_names],
                   yticklabels=[f'Sector {i}' for i in range(n_clusters)],
                   annot=True, fmt='.3f', cmap='viridis', ax=ax2)
        ax2.set_title('⚡ Firepower by Sector', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Characteristics')
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Tactical map saved: {save_path}")
    plt.show()

def plot_characteristic_distribution(
    similarity_matrices: Dict[str, np.ndarray],
    characteristic_labels: Dict[str, List[str]],
    save_path: Optional[str] = None
) -> None:
    """Plot distribution of characteristics across all images"""
    print("📊 NIMITZ: Plotting characteristic distributions...")
    
    n_chars = len(similarity_matrices)
    fig, axes = plt.subplots(2, (n_chars + 1) // 2, figsize=(15, 10))
    axes = axes.flatten() if n_chars > 1 else [axes]
    
    for i, (char_name, similarity) in enumerate(similarity_matrices.items()):
        ax = axes[i]
        
        # Plot distribution of max similarity for each image
        max_similarities = np.max(similarity, axis=1)
        ax.hist(max_similarities, bins=30, alpha=0.7, color=f'C{i}')
        ax.set_title(f'{char_name.replace("_", " ").title()}')
        ax.set_xlabel('Max Similarity Score')
        ax.set_ylabel('Number of Images')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(similarity_matrices), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Distribution plot saved: {save_path}")
    plt.show()

def plot_cluster_size_distribution(
    cluster_labels: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """Plot distribution of cluster sizes"""
    print("📊 NIMITZ: Plotting cluster size distribution...")
    
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    
    # Filter out noise (-1 in DBSCAN)
    valid_mask = unique_labels >= 0
    valid_labels = unique_labels[valid_mask]
    valid_counts = counts[valid_mask]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(valid_labels, valid_counts, color='skyblue', alpha=0.7)
    
    # Add value labels on bars
    for bar, count in zip(bars, valid_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')
    
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Images')
    plt.title('🏴‍☠️ NIMITZ - Cluster Size Distribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Cluster distribution saved: {save_path}")
    plt.show()

def create_cluster_summary_plot(
    cluster_analysis: Dict[int, Dict],
    save_path: Optional[str] = None
) -> None:
    """Create a summary visualization of all clusters"""
    print("📊 NIMITZ: Creating cluster summary visualization...")
    
    cluster_ids = list(cluster_analysis.keys())
    n_clusters = len(cluster_ids)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Cluster sizes
    sizes = [cluster_analysis[cid]['n_images'] for cid in cluster_ids]
    ax1.bar(cluster_ids, sizes, color='lightblue', alpha=0.7)
    ax1.set_title('Cluster Sizes')
    ax1.set_xlabel('Cluster ID')
    ax1.set_ylabel('Number of Images')
    
    # 2. Average characteristic scores
    all_char_types = set()
    for analysis in cluster_analysis.values():
        for char in analysis['characteristics']:
            all_char_types.add(char['characteristic'])
    
    char_scores = {char_type: [] for char_type in all_char_types}
    
    for analysis in cluster_analysis.values():
        cluster_char_scores = {char_type: 0 for char_type in all_char_types}
        for char in analysis['characteristics']:
            cluster_char_scores[char['characteristic']] = char['score']
        
        for char_type in all_char_types:
            char_scores[char_type].append(cluster_char_scores[char_type])
    
    # Plot average scores
    char_names = list(char_scores.keys())
    avg_scores = [np.mean(scores) for scores in char_scores.values()]
    
    ax2.barh(char_names, avg_scores, color='lightgreen', alpha=0.7)
    ax2.set_title('Average Characteristic Scores')
    ax2.set_xlabel('Average Score')
    
    # 3. Top characteristic per cluster
    top_chars = []
    for cid in cluster_ids:
        if cluster_analysis[cid]['characteristics']:
            top_char = cluster_analysis[cid]['characteristics'][0]['characteristic']
            top_chars.append(top_char)
        else:
            top_chars.append('None')
    
    unique_chars = list(set(top_chars))
    char_counts = [top_chars.count(char) for char in unique_chars]
    
    ax3.pie(char_counts, labels=unique_chars, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Distribution of Dominant Characteristics')
    
    # 4. Score distributions
    all_scores = []
    for analysis in cluster_analysis.values():
        for char in analysis['characteristics']:
            all_scores.append(char['score'])
    
    ax4.hist(all_scores, bins=20, color='orange', alpha=0.7)
    ax4.set_title('Distribution of Characteristic Scores')
    ax4.set_xlabel('Score')
    ax4.set_ylabel('Frequency')
    ax4.axvline(np.mean(all_scores), color='red', linestyle='--', 
                label=f'Mean: {np.mean(all_scores):.3f}')
    ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Summary plot saved: {save_path}")
    plt.show()

def plot_similarity_heatmap(
    similarity_matrices: Dict[str, np.ndarray],
    characteristic_labels: Dict[str, List[str]],
    image_paths: List,
    max_images: int = 50,
    save_path: Optional[str] = None
) -> None:
    """Plot heatmap of similarities between images and characteristics"""
    print("🔥 NIMITZ: Creating similarity heatmap...")
    
    # Limit number of images for readability
    n_images = min(len(image_paths), max_images)
    
    # Combine all similarities
    all_similarities = []
    all_labels = []
    
    for char_name, similarity in similarity_matrices.items():
        all_similarities.append(similarity[:n_images])
        for i, prompt in enumerate(characteristic_labels[char_name]):
            short_label = f"{char_name}_{i+1}"
            all_labels.append(short_label)
    
    combined_similarity = np.hstack(all_similarities)
    
    # Create heatmap
    plt.figure(figsize=(max(12, len(all_labels) * 0.5), max(8, n_images * 0.2)))
    
    sns.heatmap(combined_similarity, 
                xticklabels=all_labels,
                yticklabels=[f"Img_{i+1}" for i in range(n_images)],
                cmap='viridis', 
                cbar_kws={'label': 'Similarity Score'})
    
    plt.title('🎯 NIMITZ - Image-Characteristic Similarity Matrix')
    plt.xlabel('Characteristics')
    plt.ylabel('Images')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Similarity heatmap saved: {save_path}")
    plt.show()