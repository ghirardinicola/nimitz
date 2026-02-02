"""
Image Retrieval Module for NIMITZ
Retrieves images from the web based on text descriptions.
"""

import os
import json
import hashlib
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlencode
import time

# Try to import CLIP for image selection
try:
    import torch
    from PIL import Image
    from .embed import extract_image_features, extract_text_features
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available. Image selection will use first result.")


class ImageRetrievalError(Exception):
    """Base exception for image retrieval errors"""
    pass


class ImageSource:
    """Base class for image sources"""

    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Search for images matching the query.

        Returns list of dicts with keys:
        - url: Image URL
        - thumbnail_url: Thumbnail URL
        - title: Image title/description
        - author: Image author/creator
        - source: Source name (unsplash, bing, etc)
        - license: License information
        - attribution: Attribution text
        """
        raise NotImplementedError


class UnsplashSource(ImageSource):
    """Unsplash image source (free, high quality, good licenses)"""

    def __init__(self, access_key: Optional[str] = None):
        """
        Initialize Unsplash source.

        Args:
            access_key: Unsplash API access key. If None, uses UNSPLASH_ACCESS_KEY env var.
        """
        self.access_key = access_key or os.getenv('UNSPLASH_ACCESS_KEY')
        if not self.access_key:
            raise ImageRetrievalError(
                "Unsplash access key not provided. "
                "Set UNSPLASH_ACCESS_KEY environment variable or pass access_key parameter."
            )
        self.base_url = "https://api.unsplash.com"

    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search Unsplash for images"""
        endpoint = f"{self.base_url}/search/photos"
        params = {
            'query': query,
            'per_page': min(num_results, 30),  # Unsplash max is 30
            'client_id': self.access_key
        }

        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []
            for photo in data.get('results', []):
                results.append({
                    'url': photo['urls']['regular'],
                    'thumbnail_url': photo['urls']['thumb'],
                    'title': photo.get('description') or photo.get('alt_description') or query,
                    'author': photo['user']['name'],
                    'author_url': photo['user']['links']['html'],
                    'source': 'unsplash',
                    'source_url': photo['links']['html'],
                    'license': 'Unsplash License',
                    'license_url': 'https://unsplash.com/license',
                    'attribution': f"Photo by {photo['user']['name']} on Unsplash",
                    'width': photo['width'],
                    'height': photo['height']
                })

            return results

        except requests.exceptions.RequestException as e:
            raise ImageRetrievalError(f"Error searching Unsplash: {e}")


class PexelsSource(ImageSource):
    """Pexels image source (free, good quality, permissive license)"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Pexels source.

        Args:
            api_key: Pexels API key. If None, uses PEXELS_API_KEY env var.
        """
        self.api_key = api_key or os.getenv('PEXELS_API_KEY')
        if not self.api_key:
            raise ImageRetrievalError(
                "Pexels API key not provided. "
                "Set PEXELS_API_KEY environment variable or pass api_key parameter."
            )
        self.base_url = "https://api.pexels.com/v1"

    def search(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search Pexels for images"""
        endpoint = f"{self.base_url}/search"
        params = {
            'query': query,
            'per_page': min(num_results, 80)  # Pexels max is 80
        }
        headers = {
            'Authorization': self.api_key
        }

        try:
            response = requests.get(endpoint, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []
            for photo in data.get('photos', []):
                results.append({
                    'url': photo['src']['original'],
                    'thumbnail_url': photo['src']['medium'],
                    'title': query,  # Pexels doesn't provide titles
                    'author': photo['photographer'],
                    'author_url': photo['photographer_url'],
                    'source': 'pexels',
                    'source_url': photo['url'],
                    'license': 'Pexels License',
                    'license_url': 'https://www.pexels.com/license/',
                    'attribution': f"Photo by {photo['photographer']} on Pexels",
                    'width': photo['width'],
                    'height': photo['height']
                })

            return results

        except requests.exceptions.RequestException as e:
            raise ImageRetrievalError(f"Error searching Pexels: {e}")


def get_image_source(source_name: str) -> ImageSource:
    """
    Get an image source by name.

    Args:
        source_name: Name of the source ('unsplash', 'pexels')

    Returns:
        ImageSource instance

    Raises:
        ImageRetrievalError if source is not supported or credentials missing
    """
    sources = {
        'unsplash': UnsplashSource,
        'pexels': PexelsSource
    }

    source_class = sources.get(source_name.lower())
    if not source_class:
        raise ImageRetrievalError(
            f"Unsupported source: {source_name}. "
            f"Available sources: {', '.join(sources.keys())}"
        )

    try:
        return source_class()
    except ImageRetrievalError as e:
        raise ImageRetrievalError(f"Failed to initialize {source_name}: {e}")


def download_image(url: str, output_path: str) -> bool:
    """
    Download an image from URL to output path.

    Args:
        url: Image URL
        output_path: Path to save the image

    Returns:
        True if successful, False otherwise
    """
    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Download and save
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return True

    except Exception as e:
        print(f"Error downloading image: {e}")
        return False


def get_cache_key(description: str) -> str:
    """Generate cache key from description"""
    return hashlib.md5(description.encode()).hexdigest()


def get_cached_image(description: str, cache_dir: str) -> Optional[Dict]:
    """
    Get cached image data for a description.

    Args:
        description: Search query/description
        cache_dir: Cache directory path

    Returns:
        Dict with cached data if found, None otherwise
    """
    if not cache_dir or not os.path.exists(cache_dir):
        return None

    cache_key = get_cache_key(description)
    metadata_path = os.path.join(cache_dir, f"{cache_key}.json")

    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                data = json.load(f)

            # Check if image file exists
            if os.path.exists(data['image_path']):
                return data
        except Exception as e:
            print(f"Error reading cache: {e}")

    return None


def save_to_cache(description: str, image_path: str, metadata: Dict, cache_dir: str):
    """
    Save image and metadata to cache.

    Args:
        description: Search query/description
        image_path: Path to downloaded image
        metadata: Image metadata dict
        cache_dir: Cache directory path
    """
    if not cache_dir:
        return

    os.makedirs(cache_dir, exist_ok=True)

    cache_key = get_cache_key(description)
    metadata_path = os.path.join(cache_dir, f"{cache_key}.json")

    cache_data = {
        'description': description,
        'image_path': image_path,
        'metadata': metadata,
        'cached_at': time.time()
    }

    try:
        with open(metadata_path, 'w') as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        print(f"Error saving to cache: {e}")


def select_best_image_clip(
    image_results: List[Dict],
    description: str,
    characteristics: Dict,
    model,
    device
) -> int:
    """
    Select the best image using CLIP similarity scoring.

    Args:
        image_results: List of image result dicts with 'thumbnail_url'
        description: Original text description
        characteristics: Vocabulary characteristics to match against
        model: CLIP model
        device: Torch device

    Returns:
        Index of best matching image
    """
    if not CLIP_AVAILABLE or not image_results:
        return 0

    try:
        # Download thumbnails to temp directory
        import tempfile
        temp_dir = tempfile.mkdtemp()
        temp_paths = []

        for i, result in enumerate(image_results):
            temp_path = os.path.join(temp_dir, f"temp_{i}.jpg")
            if download_image(result['thumbnail_url'], temp_path):
                temp_paths.append(temp_path)
            else:
                temp_paths.append(None)

        # Extract image features
        valid_paths = [p for p in temp_paths if p is not None]
        if not valid_paths:
            return 0

        image_features = extract_image_features(valid_paths, model, device)

        # Create text prompts from description + characteristics
        text_prompts = [description]
        for char_name, char_data in characteristics.items():
            text_prompts.extend(char_data['prompts'][:3])  # Top 3 prompts per characteristic

        text_features = extract_text_features(text_prompts, model, device)

        # Compute similarity: average across all text prompts
        similarities = torch.matmul(image_features, text_features.T).mean(dim=1)

        # Get index of best matching image
        best_idx = similarities.argmax().item()

        # Map back to original result index (accounting for failed downloads)
        valid_idx = 0
        for i, path in enumerate(temp_paths):
            if path is not None:
                if valid_idx == best_idx:
                    return i
                valid_idx += 1

        return 0

    except Exception as e:
        print(f"Error selecting best image with CLIP: {e}")
        return 0


def retrieve_image(
    description: str,
    output_dir: str,
    source_name: str = 'unsplash',
    cache_dir: Optional[str] = None,
    characteristics: Optional[Dict] = None,
    model=None,
    device=None,
    num_candidates: int = 5
) -> Tuple[str, Dict]:
    """
    Retrieve an image from the web based on text description.

    Args:
        description: Text description/search query
        output_dir: Directory to save the image
        source_name: Image source ('unsplash', 'pexels')
        cache_dir: Optional cache directory (if None, no caching)
        characteristics: Optional characteristics dict for CLIP-based selection
        model: Optional CLIP model for image selection
        device: Optional torch device for CLIP
        num_candidates: Number of candidate images to retrieve and rank

    Returns:
        Tuple of (image_path, metadata_dict)

    Raises:
        ImageRetrievalError if retrieval fails
    """
    # Check cache first
    cached = get_cached_image(description, cache_dir) if cache_dir else None
    if cached:
        print(f"Using cached image for: {description}")
        return cached['image_path'], cached['metadata']

    # Search for images
    print(f"Searching {source_name} for: {description}")
    source = get_image_source(source_name)
    results = source.search(description, num_results=num_candidates)

    if not results:
        raise ImageRetrievalError(f"No images found for: {description}")

    # Select best image
    if characteristics and model and device and CLIP_AVAILABLE:
        print(f"Selecting best match from {len(results)} candidates...")
        best_idx = select_best_image_clip(results, description, characteristics, model, device)
    else:
        best_idx = 0

    best_result = results[best_idx]

    # Generate output filename
    safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in description[:50])
    ext = os.path.splitext(best_result['url'])[1] or '.jpg'
    output_filename = f"{safe_name}{ext}"
    output_path = os.path.join(output_dir, output_filename)

    # Download image
    print(f"Downloading image: {best_result['title']}")
    if not download_image(best_result['url'], output_path):
        raise ImageRetrievalError(f"Failed to download image from {best_result['url']}")

    # Prepare metadata
    metadata = {
        'description': description,
        'source': best_result['source'],
        'source_url': best_result['source_url'],
        'author': best_result['author'],
        'author_url': best_result.get('author_url'),
        'license': best_result['license'],
        'license_url': best_result.get('license_url'),
        'attribution': best_result['attribution'],
        'retrieved_at': time.time()
    }

    # Save to cache
    if cache_dir:
        save_to_cache(description, output_path, metadata, cache_dir)

    return output_path, metadata


def batch_retrieve_images(
    descriptions: List[str],
    output_dir: str,
    source_name: str = 'unsplash',
    cache_dir: Optional[str] = None,
    characteristics: Optional[Dict] = None,
    model=None,
    device=None,
    num_candidates: int = 5
) -> List[Tuple[str, str, Dict]]:
    """
    Retrieve multiple images from descriptions.

    Args:
        descriptions: List of text descriptions
        output_dir: Directory to save images
        source_name: Image source name
        cache_dir: Optional cache directory
        characteristics: Optional characteristics for CLIP selection
        model: Optional CLIP model
        device: Optional torch device
        num_candidates: Number of candidates per search

    Returns:
        List of tuples: (description, image_path, metadata)
    """
    results = []

    for i, description in enumerate(descriptions):
        print(f"\n[{i+1}/{len(descriptions)}] Processing: {description}")

        try:
            image_path, metadata = retrieve_image(
                description=description,
                output_dir=output_dir,
                source_name=source_name,
                cache_dir=cache_dir,
                characteristics=characteristics,
                model=model,
                device=device,
                num_candidates=num_candidates
            )
            results.append((description, image_path, metadata))
            print(f"✓ Saved: {os.path.basename(image_path)}")

        except ImageRetrievalError as e:
            print(f"✗ Failed: {e}")
            results.append((description, None, {'error': str(e)}))

        # Rate limiting (be nice to APIs)
        if i < len(descriptions) - 1:
            time.sleep(1)

    return results


def create_placeholder_image(output_path: str, text: str, size: Tuple[int, int] = (800, 600)):
    """
    Create a placeholder image with text.

    Args:
        output_path: Path to save the placeholder
        text: Text to display on placeholder
        size: Image size (width, height)
    """
    try:
        from PIL import Image, ImageDraw, ImageFont

        # Create gray background
        img = Image.new('RGB', size, color=(200, 200, 200))
        draw = ImageDraw.Draw(img)

        # Try to use a decent font, fallback to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
        except:
            font = ImageFont.load_default()

        # Draw text in center
        text_lines = [
            "Image Not Available",
            "",
            text[:40]
        ]

        y_offset = size[1] // 2 - 60
        for line in text_lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x = (size[0] - text_width) // 2
            draw.text((x, y_offset), line, fill=(100, 100, 100), font=font)
            y_offset += 40

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)

        return True

    except Exception as e:
        print(f"Error creating placeholder: {e}")
        return False


def check_api_credentials() -> Dict[str, bool]:
    """
    Check which image source APIs have credentials configured.

    Returns:
        Dict mapping source name to availability bool
    """
    sources = {}

    # Check Unsplash
    sources['unsplash'] = bool(os.getenv('UNSPLASH_ACCESS_KEY'))

    # Check Pexels
    sources['pexels'] = bool(os.getenv('PEXELS_API_KEY'))

    return sources
