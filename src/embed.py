"""
NIMITZ - Embedding Module
Feature extraction using CLIP for images and text
"""

import numpy as np
import torch
import clip
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

def initialize_clip_model(model_name: str = "ViT-B/32", device: Optional[str] = None) -> Tuple:
    """Initialize CLIP model and preprocessing"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🚢 NIMITZ deployed on: {device}")
    model, preprocess = clip.load(model_name, device=device)
    return model, preprocess, device

def extract_image_features(
    image_paths: List[Path], 
    model, 
    preprocess, 
    device: str,
    batch_size: int = 32
) -> Tuple[np.ndarray, List[Path]]:
    """Extract features from images using CLIP"""
    print(f"🔍 NIMITZ: Extracting image features...")
    
    all_features = []
    valid_paths = []
    
    # Process images in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        batch_valid_paths = []
        
        # Load and preprocess batch images
        for img_path in batch_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = preprocess(image).unsqueeze(0)
                batch_images.append(image_tensor)
                batch_valid_paths.append(img_path)
            except Exception as e:
                print(f"⚠️  Error loading {img_path}: {e}")
                continue
        
        if batch_images:
            # Stack batch images
            batch_tensor = torch.cat(batch_images, dim=0).to(device)
            
            # Extract features with CLIP
            with torch.no_grad():
                features = model.encode_image(batch_tensor)
                features = features / features.norm(dim=-1, keepdim=True)  # Normalize
                
            all_features.append(features.cpu().numpy())
            valid_paths.extend(batch_valid_paths)
    
    if all_features:
        image_features = np.vstack(all_features)
        print(f"✅ NIMITZ: Features extracted for {len(valid_paths)} images")
        return image_features, valid_paths
    else:
        raise ValueError("❌ No features extracted from images")

def extract_text_features(
    characteristics: Dict[str, List[str]], 
    model, 
    device: str,
    characteristics_to_use: Optional[List[str]] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    """Extract features for text characteristics"""
    if characteristics_to_use is None:
        characteristics_to_use = list(characteristics.keys())
    elif isinstance(characteristics_to_use, str):
        characteristics_to_use = [characteristics_to_use]
        
    print(f"🎯 NIMITZ: Extracting features for {len(characteristics_to_use)} characteristics...")
    
    characteristic_features = {}
    characteristic_labels = {}
    
    for char_name in characteristics_to_use:
        if char_name not in characteristics:
            print(f"⚠️  Characteristic '{char_name}' not found, skipping")
            continue
            
        prompts = characteristics[char_name]
        print(f"   🔍 Processing '{char_name}' ({len(prompts)} prompts)...")
        
        # Tokenize prompts for this characteristic
        text_tokens = clip.tokenize(prompts).to(device)
        
        # Extract text features
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        characteristic_features[char_name] = text_features.cpu().numpy()
        characteristic_labels[char_name] = prompts
        
    print(f"⚡ NIMITZ: {len(characteristic_features)} characteristics loaded")
    return characteristic_features, characteristic_labels

def compute_similarity_matrices(
    image_features: np.ndarray,
    characteristic_features: Dict[str, np.ndarray],
    characteristic_labels: Dict[str, List[str]],
    image_paths: List[Path]
) -> Dict[str, np.ndarray]:
    """Compute similarity matrices between images and characteristics"""
    print(f"🧭 NIMITZ: Computing similarity matrices...")
    
    similarity_matrices = {}
    
    for char_name, char_features in characteristic_features.items():
        # Compute cosine similarity for this characteristic
        similarity = np.dot(image_features, char_features.T)
        similarity_matrices[char_name] = similarity
        
    print(f"🎯 NIMITZ: Computed {len(similarity_matrices)} similarity matrices")
    return similarity_matrices

def create_combined_features(
    similarity_matrices: Dict[str, np.ndarray],
    characteristic_labels: Dict[str, List[str]],
    weighting_strategy: str = 'equal',
    custom_weights: Optional[Dict[str, float]] = None
) -> Tuple[np.ndarray, List[str]]:
    """Create combined feature space from all characteristics"""
    print(f"🧩 NIMITZ: Building unified semantic space...")
    
    # Collect all similarities
    all_similarities = []
    feature_names = []
    
    for char_name, similarity in similarity_matrices.items():
        all_similarities.append(similarity)
        
        # Create descriptive feature names
        for i, prompt in enumerate(characteristic_labels[char_name]):
            # Use abbreviated version of prompt
            short_name = prompt.split()[:3]  # First 3 words
            feature_names.append(f"{char_name}_{' '.join(short_name)}")
    
    # Concatenate all similarities
    combined_features = np.hstack(all_similarities)
    
    # Apply weighting if requested
    if weighting_strategy == 'variance':
        # Weight based on feature variance
        variances = np.var(combined_features, axis=0)
        weights = variances / np.mean(variances)
        combined_features = combined_features * weights
        print("⚖️  Applied variance-based weighting")
        
    elif weighting_strategy == 'custom' and custom_weights:
        # Apply custom weights
        char_names = list(similarity_matrices.keys())
        weights = []
        for char_name in char_names:
            char_weight = custom_weights.get(char_name, 1.0)
            n_prompts = len(characteristic_labels[char_name])
            weights.extend([char_weight] * n_prompts)
        
        weights = np.array(weights)
        combined_features = combined_features * weights
        print(f"⚖️  Applied custom weighting: {custom_weights}")
        
    print(f"🌊 NIMITZ: Unified space created - {combined_features.shape}")
    return combined_features, feature_names