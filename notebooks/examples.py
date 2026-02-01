#!/usr/bin/env python3
"""
NIMITZ - Usage Examples
Demonstrates different ways to use the refactored NIMITZ system
"""

from main import run_nimitz_pipeline, quick_analysis
from core import get_default_characteristics, save_characteristics_config

def example_1_quick_start():
    """Example 1: Quick start with default settings"""
    print("=== EXAMPLE 1: Quick Start ===")
    
    # Simple quick analysis
    results = quick_analysis(
        image_directory="./images",
        n_clusters=5,
        visualize=True
    )
    
    print(f"Found {results['n_clusters']} clusters")
    print(f"Processed {len(results['image_paths'])} images")
    return results

def example_2_custom_characteristics():
    """Example 2: Using custom characteristics"""
    print("=== EXAMPLE 2: Custom Characteristics ===")
    
    # Define custom characteristics for art analysis
    art_characteristics = {
        "art_style": [
            "realistic photographic style",
            "impressionist painting style", 
            "abstract modern art style",
            "classical renaissance style"
        ],
        "color_mood": [
            "warm and cheerful colors",
            "cool and calming colors",
            "dark and moody atmosphere",
            "bright and vibrant colors"
        ],
        "composition": [
            "centered symmetric composition",
            "dynamic diagonal composition",
            "rule of thirds composition",
            "minimalist sparse composition"
        ]
    }
    
    results = run_nimitz_pipeline(
        image_directory="./art_images",
        characteristics=art_characteristics,
        n_clusters=6,
        clustering_method='kmeans',
        visualize=True,
        output_dir="art_analysis_results"
    )
    
    return results

def example_3_photography_analysis():
    """Example 3: Photography-specific analysis"""
    print("=== EXAMPLE 3: Photography Analysis ===")
    
    photography_chars = {
        "genre": [
            "portrait photography",
            "landscape photography", 
            "street photography",
            "macro close-up photography",
            "architecture photography"
        ],
        "lighting": [
            "natural outdoor lighting",
            "studio controlled lighting",
            "golden hour warm lighting",
            "blue hour evening lighting",
            "dramatic high contrast lighting"
        ],
        "depth_of_field": [
            "shallow depth of field with bokeh",
            "deep focus everything sharp",
            "medium depth selective focus"
        ],
        "camera_angle": [
            "eye level straight angle",
            "low angle looking up",
            "high angle looking down",
            "bird's eye aerial view"
        ]
    }
    
    # Save characteristics for reuse
    save_characteristics_config(photography_chars, "photography_characteristics.json")
    
    results = run_nimitz_pipeline(
        image_directory="./photos",
        characteristics=photography_chars,
        clustering_method='dbscan',
        eps=0.3,  # DBSCAN parameter
        min_samples=3,  # DBSCAN parameter
        weighting_strategy='variance',
        visualize=True,
        output_dir="photo_analysis"
    )
    
    return results

def example_4_advanced_pipeline():
    """Example 4: Advanced pipeline with custom weights"""
    print("=== EXAMPLE 4: Advanced Analysis ===")
    
    # Load characteristics from file
    results = run_nimitz_pipeline(
        image_directory="./mixed_images",
        characteristics_file="photography_characteristics.json",
        model_name="ViT-L/14",  # Larger, more powerful model
        clustering_method='kmeans',
        n_clusters=8,
        batch_size=16,  # Smaller batch for larger model
        weighting_strategy='custom',
        custom_weights={
            'genre': 2.0,      # Weight genre more heavily
            'lighting': 1.5,   # Lighting is important
            'depth_of_field': 0.8,  # Less important
            'camera_angle': 1.0      # Normal weight
        },
        visualize=True,
        save_plots=True,
        output_dir="advanced_analysis"
    )
    
    return results

def example_5_batch_processing():
    """Example 5: Processing multiple directories"""
    print("=== EXAMPLE 5: Batch Processing ===")
    
    directories = [
        "./dataset1/images",
        "./dataset2/images", 
        "./dataset3/images"
    ]
    
    # Simple characteristics for consistent comparison
    simple_chars = {
        "content": [
            "contains people",
            "contains animals", 
            "contains buildings",
            "contains nature/landscapes",
            "contains objects/still life"
        ],
        "quality": [
            "high quality professional photo",
            "casual snapshot quality",
            "artistic creative composition"
        ]
    }
    
    all_results = []
    for i, directory in enumerate(directories):
        print(f"\nProcessing directory {i+1}: {directory}")
        
        results = run_nimitz_pipeline(
            image_directory=directory,
            characteristics=simple_chars,
            n_clusters=4,
            visualize=False,  # Skip visualization for batch
            output_dir=f"batch_results/dataset_{i+1}"
        )
        
        all_results.append(results)
    
    return all_results

def example_6_interactive_exploration():
    """Example 6: Interactive characteristic exploration"""
    print("=== EXAMPLE 6: Interactive Exploration ===")
    
    # Start with default characteristics
    base_chars = get_default_characteristics()
    
    # Run initial analysis
    initial_results = run_nimitz_pipeline(
        image_directory="./images",
        characteristics=base_chars,
        n_clusters=5,
        visualize=True,
        output_dir="exploration_v1"
    )
    
    # Based on results, refine characteristics
    refined_chars = base_chars.copy()
    refined_chars['subject_focus'] = [
        "single main subject",
        "multiple subjects",
        "abstract or pattern focus",
        "background environment focus"
    ]
    
    # Re-run with refined characteristics
    refined_results = run_nimitz_pipeline(
        image_directory="./images", 
        characteristics=refined_chars,
        n_clusters=6,
        visualize=True,
        output_dir="exploration_v2"
    )
    
    return initial_results, refined_results

def example_7_minimal_functional():
    """Example 7: Minimal functional approach"""
    print("=== EXAMPLE 7: Minimal Functional ===")
    
    # Import individual functions for custom pipeline
    from nimitz_core import load_images, get_default_characteristics
    from nimitz_embedding import initialize_clip_model, extract_image_features, extract_text_features
    from nimitz_clustering import cluster_images
    
    # Step-by-step functional approach
    characteristics = get_default_characteristics()
    image_paths = load_images("./images")
    
    model, preprocess, device = initialize_clip_model()
    image_features, valid_paths = extract_image_features(image_paths, model, preprocess, device)
    char_features, char_labels = extract_text_features(characteristics, model, device)
    
    # Simple cosine similarity without complex combining
    import numpy as np
    combined_sim = []
    for char_name, char_feat in char_features.items():
        similarity = np.dot(image_features, char_feat.T)
        combined_sim.append(similarity)
    
    combined_features = np.hstack(combined_sim)
    cluster_labels, n_clusters = cluster_images(combined_features, method='kmeans', n_clusters=4)
    
    print(f"Minimal pipeline: {len(valid_paths)} images → {n_clusters} clusters")
    return cluster_labels, valid_paths

def run_all_examples():
    """Run all examples (comment out as needed)"""
    print("🚢 NIMITZ EXAMPLES DEPLOYMENT")
    print("=" * 50)
    
    # Uncomment the examples you want to run
    # example_1_quick_start()
    # example_2_custom_characteristics() 
    # example_3_photography_analysis()
    # example_4_advanced_pipeline()
    # example_5_batch_processing()
    # example_6_interactive_exploration()
    # example_7_minimal_functional()
    
    print("\n⚓ Examples completed! Uncomment specific examples to run them.")

if __name__ == "__main__":
    run_all_examples()