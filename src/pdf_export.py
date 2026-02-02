#!/usr/bin/env python3
"""
NIMITZ - PDF Export Module
Export cards as printable PDF files
"""

import io
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image as PILImage

try:
    from reportlab.lib.pagesizes import A4, LETTER
    from reportlab.lib.units import mm, cm, inch
    from reportlab.lib.colors import HexColor, black, white, gray
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from gaming import (
    calculate_power_level,
    calculate_rarity_score,
    enhance_cards_with_gaming_stats,
    RarityTier,
    RARITY_COLORS,
    RARITY_SYMBOLS
)


# Card dimensions (standard trading card size: 63.5mm x 88.9mm)
CARD_WIDTH = 63.5 * mm
CARD_HEIGHT = 88.9 * mm
CARD_MARGIN = 3 * mm

# Page layouts
CARDS_PER_PAGE_A4 = 9   # 3x3 grid on A4
CARDS_PER_PAGE_LETTER = 9  # 3x3 grid on Letter


def check_pdf_support() -> bool:
    """Check if PDF export is available"""
    return REPORTLAB_AVAILABLE


def get_card_color(rarity_tier: str) -> str:
    """Get the border color for a rarity tier"""
    try:
        tier = RarityTier(rarity_tier)
        return RARITY_COLORS.get(tier, "#333333")
    except (ValueError, KeyError):
        return "#333333"


def create_card_image(
    card: Dict[str, Any],
    width: float,
    height: float,
    c: 'canvas.Canvas',
    x: float,
    y: float
) -> None:
    """
    Draw a single card on the canvas.

    Args:
        card: Card data
        width: Card width
        height: Card height
        c: ReportLab canvas
        x: X position
        y: Y position (bottom-left corner)
    """
    # Get rarity info
    rarity_tier = card.get('rarity_tier', 'common')
    rarity_color = get_card_color(rarity_tier)
    power_level = card.get('power_level', calculate_power_level(card))

    # Draw card background and border
    c.setStrokeColor(HexColor(rarity_color))
    c.setLineWidth(2)
    c.setFillColor(white)
    c.roundRect(x, y, width, height, 3 * mm, fill=1, stroke=1)

    # Image area (top portion)
    img_margin = 4 * mm
    img_x = x + img_margin
    img_y = y + height * 0.4
    img_width = width - 2 * img_margin
    img_height = height * 0.55

    # Try to load and draw the actual image
    image_path = card.get('image_path', '')
    if image_path and Path(image_path).exists():
        try:
            img = PILImage.open(image_path)
            img.thumbnail((int(img_width * 3), int(img_height * 3)), PILImage.Resampling.LANCZOS)

            # Convert to RGB if needed
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                background = PILImage.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            # Draw image
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='JPEG', quality=85)
            img_buffer.seek(0)
            c.drawImage(ImageReader(img_buffer), img_x, img_y, img_width, img_height,
                       preserveAspectRatio=True, anchor='c')
        except Exception:
            # Draw placeholder if image can't be loaded
            c.setFillColor(HexColor("#EEEEEE"))
            c.rect(img_x, img_y, img_width, img_height, fill=1, stroke=0)
            c.setFillColor(gray)
            c.setFont("Helvetica", 8)
            c.drawCentredString(x + width/2, img_y + img_height/2, "Image not found")
    else:
        # Draw placeholder
        c.setFillColor(HexColor("#F5F5F5"))
        c.rect(img_x, img_y, img_width, img_height, fill=1, stroke=0)

    # Card name (title bar with rarity color)
    title_height = 8 * mm
    title_y = y + height - title_height - 2 * mm
    c.setFillColor(HexColor(rarity_color))
    c.rect(x + 2 * mm, title_y, width - 4 * mm, title_height, fill=1, stroke=0)

    # Card name text
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 7)
    card_name = card.get('image_name', 'Unknown')
    if len(card_name) > 20:
        card_name = card_name[:17] + "..."
    c.drawCentredString(x + width/2, title_y + 2.5 * mm, card_name)

    # Stats section (bottom portion)
    stats_y = y + 3 * mm
    stats_height = height * 0.35 - 6 * mm

    # Power level bar
    c.setFillColor(black)
    c.setFont("Helvetica-Bold", 6)
    c.drawString(x + img_margin, stats_y + stats_height - 4 * mm, "POWER")

    # Power bar background
    bar_x = x + img_margin
    bar_y = stats_y + stats_height - 8 * mm
    bar_width = width - 2 * img_margin
    bar_height = 3 * mm

    c.setFillColor(HexColor("#E0E0E0"))
    c.rect(bar_x, bar_y, bar_width, bar_height, fill=1, stroke=0)

    # Power bar fill
    power_fill = min(1.0, power_level / 100)
    c.setFillColor(HexColor(rarity_color))
    c.rect(bar_x, bar_y, bar_width * power_fill, bar_height, fill=1, stroke=0)

    # Power value
    c.setFillColor(black)
    c.setFont("Helvetica", 6)
    c.drawRightString(x + width - img_margin, stats_y + stats_height - 4 * mm,
                     f"{power_level:.0f}")

    # Dominant features
    features = card.get('dominant_features', [])[:3]
    feature_y = bar_y - 4 * mm

    c.setFont("Helvetica", 5)
    for feature in features:
        char_name = feature.get('characteristic', '').replace('_', ' ').title()[:12]
        score = feature.get('score', 0)
        score_pct = int(score * 100)

        # Truncate if too long
        if len(char_name) > 10:
            char_name = char_name[:9] + "."

        c.setFillColor(black)
        c.drawString(bar_x, feature_y, char_name)
        c.drawRightString(x + width - img_margin, feature_y, f"{score_pct}%")

        feature_y -= 3.5 * mm

    # Rarity indicator (bottom corner)
    c.setFillColor(HexColor(rarity_color))
    c.setFont("Helvetica-Bold", 5)
    rarity_text = rarity_tier.upper()[:3]
    c.drawString(x + img_margin, y + 2 * mm, rarity_text)


def export_cards_to_pdf(
    cards: List[Dict[str, Any]],
    output_file: str,
    page_size: str = "A4",
    include_stats: bool = True
) -> Path:
    """
    Export cards to a printable PDF file.

    Cards are arranged in a 3x3 grid, suitable for printing and cutting.

    Args:
        cards: List of card dictionaries
        output_file: Output PDF file path
        page_size: Page size ('A4' or 'Letter')
        include_stats: Include gaming stats (power, rarity)

    Returns:
        Path to created PDF file
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError(
            "PDF export requires reportlab. Install with: pip install reportlab"
        )

    if not cards:
        raise ValueError("No cards to export")

    # Setup page
    if page_size.upper() == "LETTER":
        page = LETTER
        cards_per_row = 3
        cards_per_col = 3
    else:
        page = A4
        cards_per_row = 3
        cards_per_col = 3

    page_width, page_height = page
    cards_per_page = cards_per_row * cards_per_col

    # Calculate card positions
    total_card_width = CARD_WIDTH + CARD_MARGIN
    total_card_height = CARD_HEIGHT + CARD_MARGIN

    start_x = (page_width - (cards_per_row * total_card_width - CARD_MARGIN)) / 2
    start_y = page_height - (page_height - (cards_per_col * total_card_height - CARD_MARGIN)) / 2 - CARD_HEIGHT

    # Enhance cards with gaming stats if needed
    if include_stats:
        enhanced_cards = enhance_cards_with_gaming_stats(cards)
    else:
        enhanced_cards = cards

    # Create PDF
    output_path = Path(output_file)
    if not output_path.suffix:
        output_path = output_path.with_suffix('.pdf')

    c = canvas.Canvas(str(output_path), pagesize=page)

    # Draw cards
    for i, card in enumerate(enhanced_cards):
        # Calculate position on page
        page_index = i % cards_per_page
        row = page_index // cards_per_row
        col = page_index % cards_per_row

        x = start_x + col * total_card_width
        y = start_y - row * total_card_height

        # Draw card
        create_card_image(card, CARD_WIDTH, CARD_HEIGHT, c, x, y)

        # New page if needed
        if (i + 1) % cards_per_page == 0 and i + 1 < len(enhanced_cards):
            c.showPage()

    # Save PDF
    c.save()

    print(f" PDF exported: {output_path}")
    print(f"   Cards: {len(enhanced_cards)}")
    print(f"   Pages: {(len(enhanced_cards) + cards_per_page - 1) // cards_per_page}")

    return output_path


def export_single_card_pdf(
    card: Dict[str, Any],
    output_file: str,
    scale: float = 2.0
) -> Path:
    """
    Export a single card to a PDF file, scaled up for better print quality.

    Args:
        card: Card dictionary
        output_file: Output PDF file path
        scale: Scale factor (1.0 = standard trading card size)

    Returns:
        Path to created PDF file
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError(
            "PDF export requires reportlab. Install with: pip install reportlab"
        )

    # Calculate scaled dimensions
    card_width = CARD_WIDTH * scale
    card_height = CARD_HEIGHT * scale
    margin = 10 * mm

    page_width = card_width + 2 * margin
    page_height = card_height + 2 * margin

    # Enhance card with gaming stats
    enhanced_cards = enhance_cards_with_gaming_stats([card])
    enhanced_card = enhanced_cards[0]

    # Create PDF
    output_path = Path(output_file)
    if not output_path.suffix:
        output_path = output_path.with_suffix('.pdf')

    c = canvas.Canvas(str(output_path), pagesize=(page_width, page_height))

    # Draw card centered on page
    create_card_image(enhanced_card, card_width, card_height, c, margin, margin)

    c.save()

    print(f" Single card PDF exported: {output_path}")
    return output_path


def export_cards_to_png(
    cards: List[Dict[str, Any]],
    output_dir: str,
    width: int = 300,
    height: int = 420
) -> List[Path]:
    """
    Export cards as individual PNG files for printing.

    Args:
        cards: List of card dictionaries
        output_dir: Output directory
        width: Card width in pixels
        height: Card height in pixels

    Returns:
        List of created PNG file paths
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Enhance cards with gaming stats
    enhanced_cards = enhance_cards_with_gaming_stats(cards)

    created_files = []

    for card in enhanced_cards:
        # Create figure with exact dimensions
        dpi = 100
        fig_width = width / dpi
        fig_height = height / dpi
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

        # Get card info
        rarity_tier = card.get('rarity_tier', 'common')
        rarity_color = get_card_color(rarity_tier)
        power_level = card.get('power_level', 0)
        card_name = card.get('image_name', 'Unknown')

        # Card background
        rect = patches.FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.96,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=4,
            edgecolor=rarity_color,
            facecolor='white'
        )
        ax.add_patch(rect)

        # Title bar
        title_rect = patches.FancyBboxPatch(
            (0.05, 0.88), 0.9, 0.08,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=rarity_color,
            edgecolor='none'
        )
        ax.add_patch(title_rect)

        # Card name
        display_name = card_name if len(card_name) <= 20 else card_name[:17] + "..."
        ax.text(0.5, 0.92, display_name, ha='center', va='center',
               fontsize=10, fontweight='bold', color='white',
               transform=ax.transAxes)

        # Load and display image
        image_path = card.get('image_path', '')
        if image_path and Path(image_path).exists():
            try:
                img = PILImage.open(image_path)
                img.thumbnail((200, 200), PILImage.Resampling.LANCZOS)

                # Image area
                ax_img = fig.add_axes([0.1, 0.4, 0.8, 0.45])
                ax_img.imshow(img)
                ax_img.axis('off')
            except Exception:
                ax.text(0.5, 0.62, "Image not found", ha='center', va='center',
                       fontsize=8, color='gray', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.62, "No image", ha='center', va='center',
                   fontsize=8, color='gray', transform=ax.transAxes)

        # Power bar
        ax.text(0.1, 0.32, "POWER", fontsize=7, fontweight='bold',
               transform=ax.transAxes)
        ax.text(0.9, 0.32, f"{power_level:.0f}", fontsize=7, ha='right',
               transform=ax.transAxes)

        # Power bar background
        bar_bg = patches.Rectangle((0.1, 0.26), 0.8, 0.04,
                                    facecolor='#E0E0E0', edgecolor='none',
                                    transform=ax.transAxes)
        ax.add_patch(bar_bg)

        # Power bar fill
        power_fill = min(1.0, power_level / 100)
        bar_fill = patches.Rectangle((0.1, 0.26), 0.8 * power_fill, 0.04,
                                      facecolor=rarity_color, edgecolor='none',
                                      transform=ax.transAxes)
        ax.add_patch(bar_fill)

        # Dominant features
        features = card.get('dominant_features', [])[:3]
        y_pos = 0.20
        for feature in features:
            char_name = feature.get('characteristic', '').replace('_', ' ').title()
            if len(char_name) > 15:
                char_name = char_name[:14] + "."
            score = int(feature.get('score', 0) * 100)

            ax.text(0.1, y_pos, char_name, fontsize=6, transform=ax.transAxes)
            ax.text(0.9, y_pos, f"{score}%", fontsize=6, ha='right',
                   transform=ax.transAxes)
            y_pos -= 0.05

        # Rarity badge
        ax.text(0.1, 0.04, rarity_tier.upper()[:3], fontsize=6,
               fontweight='bold', color=rarity_color, transform=ax.transAxes)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Save
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in card_name)
        file_path = output_path / f"card_{safe_name}.png"
        plt.savefig(file_path, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none', pad_inches=0.1)
        plt.close()

        created_files.append(file_path)

    print(f" PNG cards exported to: {output_path}")
    print(f"   Files created: {len(created_files)}")

    return created_files


def print_pdf_export_info() -> None:
    """Print information about PDF export capabilities"""
    print("\n PDF EXPORT INFO")
    print("=" * 50)

    if REPORTLAB_AVAILABLE:
        print("  Status: Available")
        print("\n  Card size: 63.5mm x 88.9mm (standard trading card)")
        print("  Layout: 3x3 grid per page")
        print("  Supported formats: A4, Letter")
    else:
        print("  Status: NOT AVAILABLE")
        print("\n  To enable PDF export, install reportlab:")
        print("    pip install reportlab")
    print()
