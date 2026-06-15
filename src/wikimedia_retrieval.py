"""
NIMITZ - Wikipedia/Wikimedia Image Retrieval
Retrieves authentic computer scientist images from Wikipedia/Wikimedia Commons
"""

import os
import json
import requests
import hashlib
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from urllib.parse import quote
import time

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Placeholder generation disabled.")


class WikimediaRetriever:
    """Retrieve computer scientist images from Wikipedia/Wikimedia Commons"""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize Wikipedia/Wikimedia retriever

        Args:
            cache_dir: Optional cache directory for storing metadata
        """
        self.wikipedia_api = "https://en.wikipedia.org/w/api.php"
        self.commons_api = "https://commons.wikimedia.org/w/api.php"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "NIMITZ/1.0 (Educational project; https://github.com/nimitz)"
            }
        )
        self.cache_dir = cache_dir
        self.cache = {}

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def search_wikipedia_page(self, name: str) -> Optional[str]:
        """
        Search for Wikipedia page by scientist name

        Args:
            name: Scientist name (e.g., "Alan Turing")

        Returns:
            Page title if found, None otherwise
        """
        try:
            # Try exact match first
            params = {
                "action": "query",
                "titles": name,
                "format": "json",
                "redirects": 1,
            }

            response = self.session.get(self.wikipedia_api, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                if page_id != "-1":  # -1 means page not found
                    return page_data.get("title")

            # If exact match fails, try search
            search_params = {
                "action": "opensearch",
                "search": name,
                "limit": 5,
                "namespace": 0,
                "format": "json",
            }

            response = self.session.get(
                self.wikipedia_api, params=search_params, timeout=10
            )
            response.raise_for_status()
            data = response.json()

            if len(data) > 1 and len(data[1]) > 0:
                # Return first result
                return data[1][0]

            return None

        except Exception as e:
            print(f"   ⚠️  Wikipedia search failed for {name}: {e}")
            return None

    def get_page_image_url(self, page_title: str) -> Optional[Dict]:
        """
        Extract image URL from Wikipedia page

        Args:
            page_title: Wikipedia page title

        Returns:
            Dict with image info or None
        """
        try:
            # Get page images
            params = {
                "action": "query",
                "titles": page_title,
                "prop": "pageimages|images",
                "pithumbsize": 1000,  # High resolution
                "format": "json",
            }

            response = self.session.get(self.wikipedia_api, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                # Try thumbnail first (usually the main image)
                thumbnail = page_data.get("thumbnail")
                if thumbnail:
                    return {
                        "url": thumbnail["source"],
                        "width": thumbnail.get("width"),
                        "height": thumbnail.get("height"),
                        "source": "wikipedia_thumbnail",
                    }

                # Try to get original image from list
                images = page_data.get("images", [])
                if images:
                    # Look for common infobox image patterns
                    for img in images:
                        img_title = img.get("title", "")
                        # Skip icons, logos, and diagrams
                        if any(
                            skip in img_title.lower()
                            for skip in ["icon", "logo", "diagram", "signature", "flag"]
                        ):
                            continue
                        # Prefer JPG/PNG images
                        if any(
                            ext in img_title.lower()
                            for ext in [".jpg", ".jpeg", ".png"]
                        ):
                            # Get the actual file URL
                            file_url = self.get_commons_file_url(img_title)
                            if file_url:
                                return {
                                    "url": file_url,
                                    "width": None,
                                    "height": None,
                                    "source": "wikipedia_image",
                                }

            return None

        except Exception as e:
            print(f"   ⚠️  Image extraction failed for {page_title}: {e}")
            return None

    def get_commons_file_url(self, file_title: str) -> Optional[str]:
        """
        Get direct URL to file from Wikimedia Commons

        Args:
            file_title: File title (e.g., "File:Alan_Turing.jpg")

        Returns:
            Direct URL to image file
        """
        try:
            params = {
                "action": "query",
                "titles": file_title,
                "prop": "imageinfo",
                "iiprop": "url",
                "format": "json",
            }

            response = self.session.get(self.commons_api, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            for page_id, page_data in pages.items():
                imageinfo = page_data.get("imageinfo", [])
                if imageinfo and len(imageinfo) > 0:
                    return imageinfo[0].get("url")

            return None

        except Exception as e:
            print(f"   ⚠️  Commons URL retrieval failed: {e}")
            return None

    def download_image(self, url: str, output_path: str) -> bool:
        """
        Download image from URL

        Args:
            url: Image URL
            output_path: Path to save the image

        Returns:
            True if successful
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Download
            response = self.session.get(url, timeout=30, stream=True)
            response.raise_for_status()

            # Save
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return True

        except Exception as e:
            print(f"   ⚠️  Download failed: {e}")
            return False

    def create_placeholder(
        self, name: str, output_path: str, size: Tuple[int, int] = (800, 800)
    ) -> bool:
        """
        Create styled placeholder when no image found

        Args:
            name: Scientist name
            output_path: Path to save placeholder
            size: Image size (width, height)

        Returns:
            True if successful
        """
        if not PIL_AVAILABLE:
            print(f"   ⚠️  Cannot create placeholder: PIL not available")
            return False

        try:
            # Create image with gray background
            img = Image.new("RGB", size, color="#E5E5E5")
            draw = ImageDraw.Draw(img)

            # Draw border
            border_color = "#999999"
            border_width = 4
            draw.rectangle(
                [
                    (border_width, border_width),
                    (size[0] - border_width, size[1] - border_width),
                ],
                outline=border_color,
                width=border_width,
            )

            # Try to load font
            try:
                # Try common font locations
                font_paths = [
                    "/System/Library/Fonts/Helvetica.ttc",  # macOS
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # Linux
                    "C:\\Windows\\Fonts\\Arial.ttf",  # Windows
                ]
                font_large = None
                font_small = None

                for font_path in font_paths:
                    if os.path.exists(font_path):
                        font_large = ImageFont.truetype(font_path, 48)
                        font_small = ImageFont.truetype(font_path, 24)
                        break

                if font_large is None:
                    font_large = ImageFont.load_default()
                    font_small = ImageFont.load_default()

            except Exception:
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()

            # Draw scientist name (centered, multiline if needed)
            name_lines = []
            if len(name) > 20:
                # Split into multiple lines
                words = name.split()
                current_line = []
                for word in words:
                    test_line = " ".join(current_line + [word])
                    if len(test_line) <= 20:
                        current_line.append(word)
                    else:
                        if current_line:
                            name_lines.append(" ".join(current_line))
                        current_line = [word]
                if current_line:
                    name_lines.append(" ".join(current_line))
            else:
                name_lines = [name]

            # Calculate total text height
            y_offset = size[1] // 2 - (len(name_lines) * 60 + 40) // 2

            # Draw name lines
            for line in name_lines:
                bbox = draw.textbbox((0, 0), line, font=font_large)
                text_width = bbox[2] - bbox[0]
                x = (size[0] - text_width) // 2
                draw.text((x, y_offset), line, fill="#333333", font=font_large)
                y_offset += 60

            # Draw subtitle
            subtitle = "[Photo unavailable]"
            bbox = draw.textbbox((0, 0), subtitle, font=font_small)
            text_width = bbox[2] - bbox[0]
            x = (size[0] - text_width) // 2
            draw.text((x, y_offset + 20), subtitle, fill="#666666", font=font_small)

            # Save
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path)

            return True

        except Exception as e:
            print(f"   ⚠️  Placeholder creation failed: {e}")
            return False

    def get_scientist_image(self, name: str, output_dir: str) -> Dict:
        """
        Main method: Get image for a scientist

        Args:
            name: Scientist name
            output_dir: Directory to save image

        Returns:
            Dict with:
                - image_path: str
                - source: 'wikipedia' | 'placeholder'
                - attribution: str
                - license: str
                - page_url: str or None
                - success: bool
        """
        print(f"   🔍 Searching Wikipedia for: {name}")

        # Create safe filename
        safe_name = "".join(
            c if c.isalnum() or c in (" ", "_", "-") else "_" for c in name
        )
        safe_name = safe_name.replace(" ", "_")

        # Search Wikipedia
        page_title = self.search_wikipedia_page(name)

        if not page_title:
            print(f"   ❌ No Wikipedia page found")
            # Create placeholder
            output_path = os.path.join(output_dir, f"{safe_name}_placeholder.jpg")
            success = self.create_placeholder(name, output_path)

            return {
                "image_path": output_path if success else None,
                "source": "placeholder",
                "attribution": "NIMITZ Placeholder",
                "license": "N/A",
                "page_url": None,
                "success": success,
            }

        print(f"   ✓ Found page: {page_title}")

        # Get image URL
        image_info = self.get_page_image_url(page_title)

        if not image_info:
            print(f"   ❌ No image found on page")
            # Create placeholder
            output_path = os.path.join(output_dir, f"{safe_name}_placeholder.jpg")
            success = self.create_placeholder(name, output_path)

            return {
                "image_path": output_path if success else None,
                "source": "placeholder",
                "attribution": "NIMITZ Placeholder",
                "license": "N/A",
                "page_url": f"https://en.wikipedia.org/wiki/{quote(page_title)}",
                "success": success,
            }

        print(f"   ✓ Found image: {image_info['source']}")

        # Download image
        ext = os.path.splitext(image_info["url"])[1] or ".jpg"
        output_path = os.path.join(output_dir, f"{safe_name}{ext}")

        success = self.download_image(image_info["url"], output_path)

        if success:
            print(f"   ✓ Downloaded: {os.path.basename(output_path)}")
        else:
            print(f"   ❌ Download failed, creating placeholder")
            output_path = os.path.join(output_dir, f"{safe_name}_placeholder.jpg")
            success = self.create_placeholder(name, output_path)

        page_url = f"https://en.wikipedia.org/wiki/{quote(page_title)}"

        return {
            "image_path": output_path if success else None,
            "source": "wikipedia"
            if success and "placeholder" not in output_path
            else "placeholder",
            "attribution": f"Wikipedia: {page_title}"
            if success
            else "NIMITZ Placeholder",
            "license": "CC-BY-SA / Public Domain"
            if "placeholder" not in output_path
            else "N/A",
            "page_url": page_url,
            "success": success,
        }

    def batch_retrieve(
        self, names: List[str], output_dir: str, delay: float = 1.0
    ) -> List[Dict]:
        """
        Retrieve images for multiple scientists

        Args:
            names: List of scientist names
            output_dir: Directory to save images
            delay: Delay between requests (seconds) to be nice to Wikipedia

        Returns:
            List of result dicts
        """
        results = []

        for i, name in enumerate(names, 1):
            print(f"\n[{i}/{len(names)}] Processing: {name}")
            print("-" * 60)

            result = self.get_scientist_image(name, output_dir)
            result["name"] = name
            results.append(result)

            # Save metadata
            if self.cache_dir:
                cache_file = os.path.join(
                    self.cache_dir, f"{i:02d}_{name.replace(' ', '_')}.json"
                )
                with open(cache_file, "w") as f:
                    json.dump(result, f, indent=2)

            # Rate limiting
            if i < len(names):
                time.sleep(delay)

        return results


def main():
    """Test the retriever with sample scientists"""
    import sys

    print("\n" + "=" * 70)
    print("  NIMITZ - Wikipedia Image Retriever Test")
    print("=" * 70)

    # Test scientists
    test_names = [
        "Alan Turing",
        "Ada Lovelace",
        "Grace Hopper",
        "Donald Knuth",
        "Unknown Computer Scientist 12345",  # Should create placeholder
    ]

    output_dir = "./test_wiki_images"
    cache_dir = "./test_wiki_images/metadata"

    print(f"\n📁 Output directory: {output_dir}")
    print(f"📁 Cache directory: {cache_dir}\n")

    # Create retriever
    retriever = WikimediaRetriever(cache_dir=cache_dir)

    # Retrieve images
    results = retriever.batch_retrieve(test_names, output_dir, delay=0.5)

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    wikipedia_count = sum(1 for r in results if r["source"] == "wikipedia")
    placeholder_count = sum(1 for r in results if r["source"] == "placeholder")
    success_count = sum(1 for r in results if r["success"])

    print(f"\n✅ Successfully processed: {success_count}/{len(results)}")
    print(f"   Wikipedia images: {wikipedia_count}")
    print(f"   Placeholders: {placeholder_count}")

    print(f"\n📂 Files saved in: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
