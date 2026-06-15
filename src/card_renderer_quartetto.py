"""
NIMITZ - Quartetto Card Renderer
Generates Top Trumps / Quartetto style trading cards
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont, ImageColor
import io


class QuartettoCardRenderer:
    """
    Render Quartetto-style (Top Trumps) trading cards

    Card Design:
    - Header with name and card number
    - Photo section (300x300px)
    - Top skill highlight box (gold)
    - All statistics listed
    - Footer with attribution
    """

    # Card sizes
    SIZES = {
        "poker": (744, 1039),  # 63mm × 88mm @ 300 DPI (print)
        "display": (600, 850),  # For screen viewing
    }

    # Colors (Top Trumps inspired)
    COLORS = {
        "header_bg": "#1A2332",  # Navy
        "header_text": "#FFFFFF",  # White
        "top_skill_bg": "#FFD700",  # Gold
        "top_skill_text": "#000000",  # Black
        "stats_bg": "#F8F8F8",  # Very light gray
        "main_bg": "#FFFFFF",  # White
        "border": "#CCCCCC",  # Light gray
        "text": "#333333",  # Dark gray
        "text_light": "#666666",  # Medium gray
        "star": "#FFD700",  # Gold star
    }

    def __init__(self, card_size: str = "poker"):
        """
        Initialize renderer

        Args:
            card_size: 'poker' (print) or 'display' (screen)
        """
        self.size = self.SIZES.get(card_size, self.SIZES["poker"])
        self.card_size = card_size
        self.width, self.height = self.size

        # Layout dimensions (proportional to card size)
        scale = self.width / 744  # Scale from poker size

        self.layout = {
            "header_height": int(60 * scale),
            "photo_height": int(340 * scale),
            "photo_size": int(300 * scale),
            "top_skill_height": int(80 * scale),
            "stats_height": int(450 * scale),
            "footer_height": int(40 * scale),
            "margin": int(20 * scale),
            "padding": int(10 * scale),
        }

        # Font sizes (proportional)
        self.font_sizes = {
            "header_name": int(28 * scale),
            "header_number": int(18 * scale),
            "top_skill_label": int(16 * scale),
            "top_skill_value": int(24 * scale),
            "top_skill_desc": int(12 * scale),
            "stats_label": int(14 * scale),
            "stats_value": int(16 * scale),
            "footer": int(8 * scale),
        }

        # Load fonts
        self._load_fonts()

    def _load_fonts(self):
        """Load system fonts"""
        self.fonts = {}

        # Try to find system fonts
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "/System/Library/Fonts/SFNSText.ttf",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "C:\\Windows\\Fonts\\arial.ttf",  # Windows
        ]

        base_font = None
        for path in font_paths:
            if os.path.exists(path):
                base_font = path
                break

        # Load different sizes and weights
        try:
            if base_font:
                for name, size in self.font_sizes.items():
                    # Ensure minimum font size of 10
                    actual_size = max(10, size)
                    try:
                        self.fonts[name] = ImageFont.truetype(base_font, actual_size)
                        self.fonts[f"{name}_bold"] = ImageFont.truetype(
                            base_font, actual_size
                        )
                    except:
                        self.fonts[name] = ImageFont.load_default()
                        self.fonts[f"{name}_bold"] = ImageFont.load_default()
            else:
                # Fallback to default
                for name in self.font_sizes.keys():
                    self.fonts[name] = ImageFont.load_default()
                    self.fonts[f"{name}_bold"] = ImageFont.load_default()
        except Exception as e:
            print(f"   ⚠️  Font loading warning: {e}")
            # Use default fonts
            for name in self.font_sizes.keys():
                self.fonts[name] = ImageFont.load_default()
                self.fonts[f"{name}_bold"] = ImageFont.load_default()

    def _draw_text_centered(
        self,
        draw: ImageDraw,
        text: str,
        y: int,
        font: ImageFont,
        color: str,
        width: Optional[int] = None,
    ):
        """Draw centered text"""
        if width is None:
            width = self.width

        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        draw.text((x, y), text, fill=color, font=font)

    def _wrap_text(self, text: str, font: ImageFont, max_width: int) -> List[str]:
        """Wrap text to fit within max_width"""
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            test_line = " ".join(current_line + [word])
            bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox(
                (0, 0), test_line, font=font
            )
            width = bbox[2] - bbox[0]

            if width <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]

        if current_line:
            lines.append(" ".join(current_line))

        return lines

    def render_card(self, card_data: Dict, card_number: int, total_cards: int) -> Image:
        """
        Render a single card

        Args:
            card_data: Enriched card data dict
            card_number: Card number (1-based)
            total_cards: Total number of cards in deck

        Returns:
            PIL Image object
        """
        # Create base image
        img = Image.new("RGB", self.size, self.COLORS["main_bg"])
        draw = ImageDraw.Draw(img)

        y_pos = 0

        # 1. HEADER SECTION (Navy background)
        draw.rectangle(
            [(0, y_pos), (self.width, y_pos + self.layout["header_height"])],
            fill=self.COLORS["header_bg"],
        )

        # Scientist name (left)
        name = card_data["name"].upper()
        if len(name) > 25:
            # Truncate long names
            name = name[:22] + "..."

        draw.text(
            (self.layout["margin"], y_pos + self.layout["margin"]),
            name,
            fill=self.COLORS["header_text"],
            font=self.fonts["header_name_bold"],
        )

        # Card number (right)
        card_num_text = f"#{card_number}/{total_cards}"
        bbox = draw.textbbox((0, 0), card_num_text, font=self.fonts["header_number"])
        num_width = bbox[2] - bbox[0]
        draw.text(
            (
                self.width - num_width - self.layout["margin"],
                y_pos + self.layout["margin"],
            ),
            card_num_text,
            fill=self.COLORS["header_text"],
            font=self.fonts["header_number"],
        )

        y_pos += self.layout["header_height"]

        # 2. PHOTO SECTION
        photo_path = card_data.get("image")
        if photo_path and os.path.exists(photo_path):
            try:
                # Load and resize photo
                photo = Image.open(photo_path)
                photo = photo.convert("RGB")

                # Resize to fit (maintain aspect ratio)
                photo.thumbnail(
                    (self.layout["photo_size"], self.layout["photo_size"]),
                    Image.Resampling.LANCZOS,
                )

                # Center the photo
                photo_x = (self.width - photo.width) // 2
                photo_y = y_pos + (self.layout["photo_height"] - photo.height) // 2

                img.paste(photo, (photo_x, photo_y))
            except Exception as e:
                # Draw placeholder if photo fails
                draw.rectangle(
                    [
                        (self.layout["margin"], y_pos + self.layout["margin"]),
                        (
                            self.width - self.layout["margin"],
                            y_pos + self.layout["photo_height"] - self.layout["margin"],
                        ),
                    ],
                    outline=self.COLORS["border"],
                    width=2,
                )
                self._draw_text_centered(
                    draw,
                    "[Photo Error]",
                    y_pos + self.layout["photo_height"] // 2,
                    self.fonts["stats_label"],
                    self.COLORS["text_light"],
                )

        y_pos += self.layout["photo_height"]

        # 3. TOP SKILL SECTION (Gold highlight)
        top_skill = card_data.get("top_skill", {})

        draw.rectangle(
            [(0, y_pos), (self.width, y_pos + self.layout["top_skill_height"])],
            fill=self.COLORS["top_skill_bg"],
        )

        # Top skill label
        skill_y = y_pos + self.layout["padding"]
        draw.text(
            (self.layout["margin"], skill_y),
            "🏆 TOP SKILL:",
            fill=self.COLORS["top_skill_text"],
            font=self.fonts["top_skill_label_bold"],
        )

        # Top skill name and value
        skill_y += self.font_sizes["top_skill_label"] + 5
        skill_name = top_skill.get("display_name", "N/A")
        skill_value = top_skill.get("value", 0)
        skill_text = f"{skill_name}"

        draw.text(
            (self.layout["margin"], skill_y),
            skill_text,
            fill=self.COLORS["top_skill_text"],
            font=self.fonts["top_skill_value_bold"],
        )

        # Value (right aligned)
        value_text = str(skill_value)
        bbox = draw.textbbox(
            (0, 0), value_text, font=self.fonts["top_skill_value_bold"]
        )
        value_width = bbox[2] - bbox[0]
        draw.text(
            (self.width - value_width - self.layout["margin"], skill_y),
            value_text,
            fill=self.COLORS["top_skill_text"],
            font=self.fonts["top_skill_value_bold"],
        )

        # Description (wrapped, italic style - we'll use regular font but smaller)
        skill_desc = top_skill.get("description", "")
        if skill_desc:
            skill_y += self.font_sizes["top_skill_value"] + 3
            desc_wrapped = self._wrap_text(
                f'"{skill_desc}"',
                self.fonts["top_skill_desc"],
                self.width - 2 * self.layout["margin"],
            )
            for line in desc_wrapped[:2]:  # Max 2 lines
                draw.text(
                    (self.layout["margin"], skill_y),
                    line,
                    fill=self.COLORS["top_skill_text"],
                    font=self.fonts["top_skill_desc"],
                )
                skill_y += self.font_sizes["top_skill_desc"] + 2

        y_pos += self.layout["top_skill_height"]

        # 4. STATS SECTION
        draw.rectangle(
            [(0, y_pos), (self.width, y_pos + self.layout["stats_height"])],
            fill=self.COLORS["stats_bg"],
        )

        # Stats header
        stats_y = y_pos + self.layout["padding"]
        draw.text(
            (self.layout["margin"], stats_y),
            "ALL STATISTICS:",
            fill=self.COLORS["text"],
            font=self.fonts["stats_label_bold"],
        )

        stats_y += self.font_sizes["stats_label"] + self.layout["padding"]

        # Draw all statistics
        scores = card_data.get("scores", {})
        top_skill_key = top_skill.get("key")

        # Sort by rank
        sorted_scores = sorted(scores.items(), key=lambda x: x[1].get("rank", 99))

        for char_key, char_data in sorted_scores:
            display_name = char_data.get("display_name", char_key)
            value = char_data.get("value", 0)

            # Stat name
            draw.text(
                (self.layout["margin"] + 5, stats_y),
                display_name,
                fill=self.COLORS["text"],
                font=self.fonts["stats_label"],
            )

            # Stat value (right aligned)
            value_text = str(value)
            if char_key == top_skill_key:
                value_text += " ★"  # Star for best stat

            bbox = draw.textbbox(
                (0, 0), value_text, font=self.fonts["stats_value_bold"]
            )
            value_width = bbox[2] - bbox[0]
            draw.text(
                (self.width - value_width - self.layout["margin"], stats_y),
                value_text,
                fill=self.COLORS["text"],
                font=self.fonts["stats_value_bold"],
            )

            stats_y += self.font_sizes["stats_value"] + 8

        y_pos += self.layout["stats_height"]

        # 5. FOOTER SECTION
        # Attribution (left)
        attribution = card_data.get("attribution", "Unknown Source")
        draw.text(
            (self.layout["margin"], y_pos + self.layout["padding"]),
            attribution,
            fill=self.COLORS["text_light"],
            font=self.fonts["footer"],
        )

        # Card number (right)
        footer_text = f"Card {card_number}/{total_cards}"
        bbox = draw.textbbox((0, 0), footer_text, font=self.fonts["footer"])
        footer_width = bbox[2] - bbox[0]
        draw.text(
            (
                self.width - footer_width - self.layout["margin"],
                y_pos + self.layout["padding"],
            ),
            footer_text,
            fill=self.COLORS["text_light"],
            font=self.fonts["footer"],
        )

        return img

    def render_card_back(self) -> Image:
        """
        Render unified card back with NIMITZ logo

        Returns:
            PIL Image object
        """
        img = Image.new("RGB", self.size, self.COLORS["header_bg"])
        draw = ImageDraw.Draw(img)

        # Create gradient effect (simple: darker at top, lighter at bottom)
        # We'll skip complex gradient for simplicity

        # NIMITZ text (large, centered)
        nimitz_text = "N I M I T Z"
        font_large = self.fonts.get("header_name_bold", ImageFont.load_default())

        # Scale up the font for back
        try:
            font_paths = [
                path
                for path in [
                    "/System/Library/Fonts/Helvetica.ttc",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                ]
                if os.path.exists(path)
            ]

            if font_paths:
                font_large = ImageFont.truetype(
                    font_paths[0], int(48 * self.width / 744)
                )
                font_small = ImageFont.truetype(
                    font_paths[0], int(20 * self.width / 744)
                )
                font_tiny = ImageFont.truetype(
                    font_paths[0], int(14 * self.width / 744)
                )
            else:
                font_small = font_large
                font_tiny = font_large
        except:
            font_small = font_large
            font_tiny = font_large

        # Center position
        center_y = self.height // 2

        # NIMITZ
        self._draw_text_centered(
            draw, nimitz_text, center_y - 80, font_large, self.COLORS["header_text"]
        )

        # Subtitle
        self._draw_text_centered(
            draw,
            "Computer Scientist Cards",
            center_y - 10,
            font_small,
            self.COLORS["header_text"],
        )

        # Anchor symbol
        self._draw_text_centered(
            draw, "⚓", center_y + 40, font_large, self.COLORS["star"]
        )

        # Quartetto style
        self._draw_text_centered(
            draw,
            "Top Trumps Style",
            center_y + 120,
            font_tiny,
            self.COLORS["text_light"],
        )

        # Border
        draw.rectangle(
            [(5, 5), (self.width - 5, self.height - 5)],
            outline=self.COLORS["star"],
            width=3,
        )

        return img

    def save_card(self, image: Image, output_path: str):
        """Save card as PNG"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path, "PNG", quality=95)


def generate_all_cards(
    json_file: str,
    output_dir: str,
    card_size: str = "poker",
    generate_backs: bool = True,
):
    """
    Generate all cards from enriched JSON

    Args:
        json_file: Path to enriched JSON file
        output_dir: Output directory
        card_size: 'poker' or 'display'
        generate_backs: Whether to generate card backs
    """
    import json

    print(f"\n🎨 Generating {card_size} cards...")

    # Load data
    with open(json_file) as f:
        cards_data = json.load(f)

    print(f"✓ Loaded {len(cards_data)} cards")

    # Create renderer
    renderer = QuartettoCardRenderer(card_size=card_size)

    # Create output directory
    fronts_dir = os.path.join(output_dir, card_size, "fronts")
    os.makedirs(fronts_dir, exist_ok=True)

    # Render each card
    for i, card_data in enumerate(cards_data, 1):
        name = card_data["name"]
        safe_name = "".join(
            c if c.isalnum() or c in (" ", "_", "-") else "_" for c in name
        )
        safe_name = safe_name.replace(" ", "_")

        print(f"   [{i}/{len(cards_data)}] Rendering: {name}")

        # Render front
        card_img = renderer.render_card(card_data, i, len(cards_data))

        # Save front
        front_path = os.path.join(fronts_dir, f"{i:02d}_{safe_name}.png")
        renderer.save_card(card_img, front_path)

    # Generate card back
    if generate_backs:
        backs_dir = os.path.join(output_dir, card_size, "backs")
        os.makedirs(backs_dir, exist_ok=True)

        print(f"\n   Generating card back...")
        back_img = renderer.render_card_back()
        back_path = os.path.join(backs_dir, "card_back.png")
        renderer.save_card(back_img, back_path)

    print(f"\n✅ Generated {len(cards_data)} cards in: {fronts_dir}")
    if generate_backs:
        print(f"✅ Card back saved in: {backs_dir}")


def render_grid_page(
    card_images: List[Image.Image],
    page_size: Tuple[int, int] = (2480, 3508),  # A4 @ 300 DPI
    grid_size: Tuple[int, int] = (3, 3),  # 3×3 grid
    margin: int = 60,  # Margin around page
    spacing: int = 30,  # Space between cards
    cut_marks: bool = True,
) -> Image.Image:
    """
    Render a grid page with multiple cards for printing

    Args:
        card_images: List of card images (up to 9 for 3×3)
        page_size: Page dimensions (width, height) in pixels
        grid_size: Grid dimensions (cols, rows)
        margin: Margin around page edges
        spacing: Space between cards
        cut_marks: Whether to add cut marks

    Returns:
        Image of the grid page
    """
    page_width, page_height = page_size
    cols, rows = grid_size

    # Create white page
    page = Image.new("RGB", page_size, "white")
    draw = ImageDraw.Draw(page)

    # Calculate card positions
    available_width = page_width - (2 * margin) - ((cols - 1) * spacing)
    available_height = page_height - (2 * margin) - ((rows - 1) * spacing)

    card_width = available_width // cols
    card_height = available_height // rows

    # Place cards
    card_idx = 0
    for row in range(rows):
        for col in range(cols):
            if card_idx >= len(card_images):
                break

            # Calculate position
            x = margin + (col * (card_width + spacing))
            y = margin + (row * (card_height + spacing))

            # Resize card to fit grid cell
            card = card_images[card_idx].copy()
            card = card.resize((card_width, card_height), Image.Resampling.LANCZOS)

            # Paste card
            page.paste(card, (x, y))

            # Add cut marks if requested
            if cut_marks:
                mark_length = 20
                mark_color = "#999999"

                # Top-left
                draw.line(
                    [(x - 10, y), (x - 10 - mark_length, y)], fill=mark_color, width=1
                )
                draw.line(
                    [(x, y - 10), (x, y - 10 - mark_length)], fill=mark_color, width=1
                )

                # Top-right
                draw.line(
                    [(x + card_width + 10, y), (x + card_width + 10 + mark_length, y)],
                    fill=mark_color,
                    width=1,
                )
                draw.line(
                    [(x + card_width, y - 10), (x + card_width, y - 10 - mark_length)],
                    fill=mark_color,
                    width=1,
                )

                # Bottom-left
                draw.line(
                    [
                        (x - 10, y + card_height),
                        (x - 10 - mark_length, y + card_height),
                    ],
                    fill=mark_color,
                    width=1,
                )
                draw.line(
                    [
                        (x, y + card_height + 10),
                        (x, y + card_height + 10 + mark_length),
                    ],
                    fill=mark_color,
                    width=1,
                )

                # Bottom-right
                draw.line(
                    [
                        (x + card_width + 10, y + card_height),
                        (x + card_width + 10 + mark_length, y + card_height),
                    ],
                    fill=mark_color,
                    width=1,
                )
                draw.line(
                    [
                        (x + card_width, y + card_height + 10),
                        (x + card_width, y + card_height + 10 + mark_length),
                    ],
                    fill=mark_color,
                    width=1,
                )

            card_idx += 1

    return page


def generate_print_pages(
    json_file: str,
    output_dir: str,
    card_size: str = "poker",
    cards_per_page: int = 9,
    generate_backs: bool = True,
) -> List[str]:
    """
    Generate printable grid pages from card data

    Args:
        json_file: Path to enriched JSON file
        output_dir: Output directory
        card_size: Card size ('poker' or 'display')
        cards_per_page: Number of cards per page (9 for 3×3)
        generate_backs: Whether to generate back pages

    Returns:
        List of generated page file paths
    """
    import json

    print(f"\n🖨️  Generating print pages...")

    # Load data
    with open(json_file) as f:
        cards_data = json.load(f)

    print(f"✓ Loaded {len(cards_data)} cards")

    # Create renderer
    renderer = QuartettoCardRenderer(card_size)

    # Render all cards
    print(f"   Rendering {len(cards_data)} cards...")
    card_images = []
    for i, card_data in enumerate(cards_data, 1):
        card_img = renderer.render_card(
            card_data, card_number=i, total_cards=len(cards_data)
        )
        card_images.append(card_img)

    # Split into pages
    pages = []
    page_images = []

    for i in range(0, len(card_images), cards_per_page):
        page_cards = card_images[i : i + cards_per_page]
        page_img = render_grid_page(page_cards)
        page_images.append(page_img)
        pages.append(f"page_{(i // cards_per_page) + 1:02d}_fronts.png")

    # Save pages
    print_dir = os.path.join(output_dir, card_size, "print_pages")
    os.makedirs(print_dir, exist_ok=True)

    saved_paths = []
    for i, (page_img, page_name) in enumerate(zip(page_images, pages), 1):
        page_path = os.path.join(print_dir, page_name)
        page_img.save(page_path, dpi=(300, 300))
        print(f"   ✓ Saved page {i}/{len(pages)}: {page_name}")
        saved_paths.append(page_path)

    # Generate back pages if requested
    if generate_backs:
        print(f"\n   Generating card back pages...")
        card_back = renderer.render_card_back()
        back_images = [card_back] * cards_per_page

        num_back_pages = len(page_images)
        for i in range(num_back_pages):
            back_page = render_grid_page(back_images)
            back_page_name = f"page_{i + 1:02d}_backs.png"
            back_page_path = os.path.join(print_dir, back_page_name)
            back_page.save(back_page_path, dpi=(300, 300))
            print(f"   ✓ Saved back page {i + 1}/{num_back_pages}: {back_page_name}")
            saved_paths.append(back_page_path)

    print(f"\n✅ Generated {len(saved_paths)} print pages in: {print_dir}")
    return saved_paths


def export_to_pdf(
    json_file: str,
    output_path: str,
    card_size: str = "poker",
    include_backs: bool = True,
):
    """
    Export cards to print-ready PDF

    Args:
        json_file: Path to enriched JSON file
        output_path: Output PDF file path
        card_size: Card size ('poker' or 'display')
        include_backs: Whether to include back pages
    """
    try:
        from PIL import Image
        import img2pdf
    except ImportError:
        print("❌ img2pdf not installed. Install with: pip install img2pdf")
        return

    print(f"\n📄 Generating PDF: {output_path}")

    # Generate print pages
    page_paths = generate_print_pages(
        json_file=json_file,
        output_dir=os.path.dirname(output_path) or ".",
        card_size=card_size,
        generate_backs=include_backs,
    )

    # Sort pages: all fronts, then all backs
    front_pages = sorted([p for p in page_paths if "fronts" in p])
    back_pages = sorted([p for p in page_paths if "backs" in p])

    # Combine in order
    all_pages = front_pages + back_pages

    # Convert to PDF
    print(f"\n   Converting {len(all_pages)} pages to PDF...")
    with open(output_path, "wb") as f:
        f.write(img2pdf.convert(all_pages))

    print(f"✅ PDF saved: {output_path}")


if __name__ == "__main__":
    # Test with enriched JSON
    generate_all_cards(
        json_file="informatici_cards_analysis_enriched.json",
        output_dir="informatici_cards_quartetto",
        card_size="display",
        generate_backs=True,
    )
