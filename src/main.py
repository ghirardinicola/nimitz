#!/usr/bin/env python3
"""
NIMITZ - Main Pipeline
Functional approach to semantic image clustering
"""

from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path

# Import all modules
from image_card import (
    generate_image_cards_data,
    print_simple_image_cards,
    print_all_feature_values,
    create_visual_image_cards,
    export_cards_to_csv,
    find_similar_images
)

from core import (
    load_characteristics_from_json, 
    get_default_characteristics,
    load_images,
    validate_characteristics,
    print_characteristics_summary,
    save_results
)

from embed import (
    initialize_clip_model,
    extract_image_features,
    extract_text_features,
    compute_similarity_matrices,
    create_combined_features
)

from cluster import (
    cluster_images,
    create_results_dataframe,
    analyze_cluster_characteristics,
    discover_characteristic_combinations,
    generate_cluster_cards
)

from viz import (
    visualize_clusters,
    plot_characteristic_distribution,
    plot_cluster_size_distribution,
    create_cluster_summary_plot,
    plot_similarity_heatmap
)

def run_nimitz_pipeline(
    image_directory: str,
    characteristics: Optional[Dict[str, List[str]]] = None,
    characteristics_file: Optional[str] = None,
    model_name: str = "ViT-B/32",
    clustering_method: str = 'kmeans',
    n_clusters: int = 5,
    batch_size: int = 32,
    weighting_strategy: str = 'equal',
    custom_weights: Optional[Dict[str, float]] = None,
    output_dir: str = "../results",
    visualize: bool = True,
    save_plots: bool = True,
    **clustering_kwargs
) -> Dict[str, Any]:
    """
    Complete NIMITZ pipeline for semantic image clustering
    
    Args:
        image_directory: Path to directory containing images
        characteristics: Dict of characteristics {name: [prompts]}
        characteristics_file: Path to JSON file with characteristics
        model_name: CLIP model to use
        clustering_method: 'kmeans' or 'dbscan'
        n_clusters: Number of clusters (for kmeans)
        batch_size: Batch size for image processing
        weighting_strategy: 'equal', 'variance', or 'custom'
        custom_weights: Custom weights for characteristics
        output_dir: Directory to save results
        visualize: Whether to show visualizations
        save_plots: Whether to save plot files
        **clustering_kwargs: Additional clustering parameters
    
    Returns:
        Dict containing all results and data
    """
    
    print("🚢 NIMITZ PIPELINE INITIATED")
    print("=" * 50)
    
    # 1. Setup characteristics
    if characteristics_file:
        characteristics = load_characteristics_from_json(characteristics_file)
    elif characteristics is None:
        characteristics = get_default_characteristics()
        
    if not validate_characteristics(characteristics):
        raise ValueError("Invalid characteristics provided")
    
    print_characteristics_summary(characteristics)
    
    # 2. Initialize CLIP model
    model, preprocess, device = initialize_clip_model(model_name)
    
    # 3. Load and process images
    image_paths = load_images(image_directory)
    if not image_paths:
        raise ValueError("No images found in directory")
    
    # 4. Extract features
    image_features, valid_image_paths = extract_image_features(
        image_paths, model, preprocess, device, batch_size
    )
    
    characteristic_features, characteristic_labels = extract_text_features(
        characteristics, model, device
    )
    
    # 5. Compute similarities
    similarity_matrices = compute_similarity_matrices(
        image_features, characteristic_features, characteristic_labels, valid_image_paths
    )
    
    # 6. Create combined feature space
    combined_features, feature_names = create_combined_features(
        similarity_matrices, characteristic_labels, weighting_strategy, custom_weights
    )
    
     # 11. Generate image cards
    image_cards_data = generate_image_cards_data(
        valid_image_paths, similarity_matrices, characteristic_labels
    )
        
    # Print cards to console
    print_simple_image_cards(
        image_cards_data, 
        max_cards=5
    )
            
    # Also print raw feature values if requested
    print("\n" + "="*100)
    print("📊 RAW FEATURE VALUES FOR ALL IMAGES")
    print("="*100)
    print_all_feature_values(
        valid_image_paths,
        similarity_matrices,
        characteristic_labels,
        max_images=5
    )
        
    # Create visual cards
    create_visual_image_cards(
        image_cards_data, 
        output_dir=f"{output_dir}/visual_cards" if output_dir else "visual_cards"
    )
    
    # Export to CSV
    export_cards_to_csv(
        image_cards_data,
        output_file=f"{output_dir}/image_cards.csv" if output_dir else "image_cards.csv"
    )

        # 7. Cluster images
    cluster_labels, final_n_clusters = cluster_images(
        combined_features, clustering_method, n_clusters, **clustering_kwargs
    )
    
    # 8. Create results dataframe
    results_df = create_results_dataframe(valid_image_paths, cluster_labels)
    
    # 9. Analyze clusters
    cluster_analysis = analyze_cluster_characteristics(
        cluster_labels, combined_features, feature_names, 
        characteristic_labels, final_n_clusters, results_df
    )

    
    # 10. Discover combinations
    discovered_combinations = discover_characteristic_combinations(cluster_analysis)
    
    # 11. Generate cluster cards
    cluster_cards = generate_cluster_cards(
        cluster_analysis, results_df, cluster_labels, combined_features
    )
    
    # 12. Visualizations
    if visualize:

        # Visualize one card
        print("sample card")
        print(cluster_cards[0])

        # Main cluster visualization
        visualize_clusters(
            combined_features, cluster_labels, final_n_clusters,
            cluster_analysis, characteristics,
            save_path=f"{output_dir}/cluster_map.png" if save_plots else None
        )
        
        # Additional plots
        plot_characteristic_distribution(
            similarity_matrices, characteristic_labels,
            save_path=f"{output_dir}/char_distribution.png" if save_plots else None
        )
        
        plot_cluster_size_distribution(
            cluster_labels,
            save_path=f"{output_dir}/cluster_sizes.png" if save_plots else None
        )
        
        create_cluster_summary_plot(
            cluster_analysis,
            save_path=f"{output_dir}/cluster_summary.png" if save_plots else None
        )
        
        plot_similarity_heatmap(
            similarity_matrices, characteristic_labels, valid_image_paths,
            save_path=f"{output_dir}/similarity_heatmap.png" if save_plots else None
        )
    
    # 13. Prepare results for saving
    results_data = {
        "clustering_results.csv": results_df,
        "cluster_analysis.json": cluster_analysis,
        "discovered_combinations.json": discovered_combinations,
        "cluster_cards.json": cluster_cards,
        "characteristics_config.json": characteristics,
        "mission_summary.json": {
            'mission_summary': {
                'total_images': len(valid_image_paths),
                'total_clusters': final_n_clusters,
                'characteristics_used': list(characteristics.keys()),
                'total_prompts': sum(len(prompts) for prompts in characteristics.values()),
                'combinations_discovered': len(discovered_combinations),
                'clustering_method': clustering_method,
                'weighting_strategy': weighting_strategy
            }
        }
    }
    
    # 14. Save results
    output_path = save_results(results_data, output_dir)
    
    print("\n🎯 NIMITZ MISSION COMPLETED")
    print("=" * 50)
    print(f"📊 Processed: {len(valid_image_paths)} images")
    print(f"🎲 Created: {final_n_clusters} clusters")
    print(f"🌟 Discovered: {len(discovered_combinations)} combinations")
    print(f"📁 Results saved to: {output_path}")
    
    return {
        'image_paths': valid_image_paths,
        'cluster_labels': cluster_labels,
        'n_clusters': final_n_clusters,
        'results_df': results_df,
        'cluster_analysis': cluster_analysis,
        'discovered_combinations': discovered_combinations,
        'cluster_cards': cluster_cards,
        'characteristics': characteristics,
        'similarity_matrices': similarity_matrices,
        'combined_features': combined_features,
        'feature_names': feature_names,
        'output_path': output_path
    }

def quick_analysis(
    image_directory: str,
    characteristics_config: Optional[str] = None,
    n_clusters: int = 3,
    visualize: bool = True
) -> Dict[str, Any]:
    """
    Quick analysis function for fast clustering
    
    Args:
        image_directory: Directory with images
        characteristics_config: Path to characteristics JSON or None for defaults
        n_clusters: Number of clusters
        visualize: Whether to show plots
    
    Returns:
        Results dictionary
    """
    print("🚢 NIMITZ: Quick mission initiated!")
    
    return run_nimitz_pipeline(
        image_directory=image_directory,
        characteristics_file=characteristics_config,
        n_clusters=n_clusters,
        visualize=visualize,
        save_plots=visualize
    )

# Utility functions for interactive use
def create_custom_characteristics() -> Dict[str, List[str]]:
    """Helper to create custom characteristics interactively"""
    print("🎯 NIMITZ: Creating custom characteristics arsenal")
    characteristics = {}
    
    while True:
        char_name = input("\nEnter characteristic name (or 'done' to finish): ").strip()
        if char_name.lower() == 'done':
            break
            
        prompts = []
        print(f"Enter prompts for '{char_name}' (empty line to finish):")
        while True:
            prompt = input("  > ").strip()
            if not prompt:
                break
            prompts.append(prompt)
        
        if prompts:
            characteristics[char_name] = prompts
            print(f"✅ Added '{char_name}' with {len(prompts)} prompts")
    
    return characteristics

# Example usage and testing
if __name__ == "__main__":
    # Example: Quick analysis with default characteristics
    # results = quick_analysis("path/to/images", n_clusters=5)
    
    # Example: Custom characteristics
    # custom_chars = {
    #     "style": ["realistic photography", "artistic painting", "cartoon illustration"],
    #     "subject": ["people", "animals", "landscapes", "objects"]
    # }
    # results = run_nimitz_pipeline("path/to/images", characteristics=custom_chars)
    
    print("🚢 NIMITZ ready for deployment!")
    print("Use quick_analysis() or run_nimitz_pipeline() to start clustering")