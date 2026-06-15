# NIMITZ Quartetto Cards - Computer Scientist Trading Cards

A complete system for generating professional Top Trumps/Quartetto-style trading cards featuring famous computer scientists with authentic Wikipedia images.

## 🎴 Card Design

The cards feature a **Top Trumps/Quartetto** layout with:
- **Header**: Scientist name + card number
- **Photo**: 300×300px Wikipedia image (or placeholder)
- **Top Skill Highlight**: Gold box showing their strongest characteristic
- **Stats Section**: All 5 characteristics ranked with values
- **Footer**: Wikipedia attribution

### Card Sizes
- **Display**: 600×850 pixels (for screen viewing)
- **Poker**: 744×1039 pixels (63mm×88mm @ 300 DPI, print-ready)

### Characteristics (0-100 scale)
1. **Market Influence** - Impact on the tech industry
2. **Advanced Technologies** - Mastery of cutting-edge tech
3. **Professional Recognition** - Industry reputation
4. **Beard Length** - Facial hair magnificence 🧔
5. **Openness** - Open source contributions

Each characteristic has 5 description levels based on the score range (0-20, 21-40, 41-60, 61-80, 81-100).

## 📁 Project Structure

```
nimitz/
├── informatici.txt                                 # List of 32 scientists
├── informatici_cards_analysis.json                 # Original analysis with scores
├── informatici_cards_analysis_enriched.json        # Enriched with descriptions
├── vocabolario_informatici_game_en.json            # English vocabulary/descriptions
├── cards/quartetto/wiki_images/                         # Wikipedia images (auto-downloaded)
│   ├── *.png, *.jpg                                # 32 scientist photos
│   └── metadata/                                   # Attribution info
├── cards/quartetto/                    # Generated cards
│   ├── display/                                    # 600×850 cards
│   │   ├── fronts/                                 # 32 card fronts
│   │   └── backs/                                  # Card back design
│   └── poker/                                      # 744×1039 cards (print)
│       ├── fronts/                                 # 32 card fronts
│       ├── backs/                                  # Card back design
│       ├── print_pages/                            # 3×3 grids on A4
│       │   ├── page_01_fronts.png                  # 9 cards per page
│       │   ├── page_01_backs.png
│       │   └── ...
│       └── NIMITZ_Cards_Printable.pdf              # Print-ready PDF
├── src/
│   ├── wikimedia_retrieval.py                      # Wikipedia image downloader
│   └── card_renderer_quartetto.py                  # Card rendering engine
├── enrich_cards_json.py                            # Add descriptions to JSON
└── generate_quartetto_full.py                      # Complete pipeline script
```

## 🚀 Quick Start

### One-Command Generation

```bash
python3 generate_quartetto_full.py
```

This will:
1. Download Wikipedia images for all 32 scientists
2. Enrich JSON data with descriptions
3. Generate display cards (600×850)
4. Generate poker cards (744×1039)
5. Create 3×3 print pages on A4
6. Export print-ready PDF

### Step-by-Step Generation

```bash
# 1. Enrich JSON with English descriptions
python3 enrich_cards_json.py

# 2. Generate display cards
python3 -c "from src.card_renderer_quartetto import generate_all_cards; \
generate_all_cards('informatici_cards_analysis_enriched.json', 'cards/quartetto', 'display', True)"

# 3. Generate poker (print) cards
python3 -c "from src.card_renderer_quartetto import generate_all_cards; \
generate_all_cards('informatici_cards_analysis_enriched.json', 'cards/quartetto', 'poker', True)"

# 4. Generate print pages
python3 -c "from src.card_renderer_quartetto import generate_print_pages; \
generate_print_pages('informatici_cards_analysis_enriched.json', 'cards/quartetto', 'poker', 9, True)"

# 5. Generate PDF
python3 -c "from src.card_renderer_quartetto import export_to_pdf; \
export_to_pdf('informatici_cards_analysis_enriched.json', 'cards/quartetto/poker/NIMITZ_Cards_Printable.pdf', 'poker', True)"
```

## 📦 Dependencies

```bash
pip install pillow requests img2pdf
```

## 🖨️ Printing Instructions

1. **Open PDF**: `cards/quartetto/poker/NIMITZ_Cards_Printable.pdf`
2. **Print Settings**:
   - Paper: A4 (210×297mm)
   - Quality: 300 DPI minimum
   - Color: Yes
   - Double-sided: Optional
3. **Cutting**:
   - Each page has 9 cards (3×3 grid)
   - Cut marks shown at corners
   - Card size: 63×88mm (standard poker size)
4. **Optional**:
   - Laminate cards for durability
   - Print on cardstock (200-300gsm recommended)

## 👥 Scientists Included (32 total)

- Ada Lovelace - First programmer
- Alan Turing - Computing pioneer
- Grace Hopper - COBOL creator
- Donald Knuth - TeX & algorithms
- Tim Berners-Lee - Web inventor
- Linus Torvalds - Linux creator
- Guido van Rossum - Python creator
- Bjarne Stroustrup - C++ creator
- Dennis Ritchie - C language
- And 23 more legendary computer scientists!

## 🎨 Customization

### Modify Characteristics

Edit `vocabolario_informatici_game_en.json`:
```json
{
  "characteristics": {
    "market_influence": [
      "Level 0 description (0-20)",
      "Level 1 description (21-40)",
      ...
    ]
  },
  "display_names": {
    "market_influence": "Market Influence"
  }
}
```

### Change Card Colors

Edit `src/card_renderer_quartetto.py`:
```python
COLORS = {
    "header_bg": "#1A2332",  # Navy header
    "top_skill_bg": "#FFD700",  # Gold highlight
    ...
}
```

### Add More Scientists

1. Add name to `informatici.txt`
2. Add scores to `informatici_cards_analysis.json`
3. Run `python3 generate_quartetto_full.py`

## 📝 Data Format

### Enriched JSON Structure

```json
{
  "name": "Ada Lovelace",
  "image": "./cards/quartetto/wiki_images/Ada_Lovelace.png",
  "image_source": "wikipedia",
  "attribution": "Wikimedia Commons",
  "page_url": "https://en.wikipedia.org/wiki/Ada_Lovelace",
  "scores": {
    "market_influence": {
      "value": 74,
      "level": 3,
      "description": "developer with great impact on a technology sector",
      "rank": 2,
      "display_name": "Market Influence"
    }
    ...
  },
  "top_skill": {
    "key": "advanced_technologies",
    "display_name": "Advanced Tech",
    "value": 75,
    "description": "developer with advanced skills in emerging technologies..."
  }
}
```

## 🔧 Technical Details

### Wikipedia Image Download
- Uses Wikipedia API to search for scientist pages
- Extracts infobox images from Wikimedia Commons
- Downloads high-resolution originals
- Creates styled placeholders when images unavailable
- Stores attribution metadata in JSON

### Card Rendering
- Uses PIL (Pillow) for image processing
- System fonts (Helvetica/Arial/DejaVu)
- 300 DPI for print quality
- Anti-aliased text rendering
- Centered photo cropping

### Print Layout
- A4 pages at 2480×3508 pixels (300 DPI)
- 3×3 grid with margins and spacing
- Cut marks at card corners
- PDF uses img2pdf for lossless conversion

## 📊 Output Statistics

- **Total cards**: 32 unique scientists
- **Wikipedia images**: ~30 (varies by availability)
- **Placeholder images**: ~2 (scientists without photos)
- **Display cards**: 32 fronts + 1 back
- **Poker cards**: 32 fronts + 1 back
- **Print pages**: 4 fronts + 4 backs
- **PDF pages**: 8 total (4 fronts, 4 backs)

## 🎯 Use Cases

- Educational tool for computer science history
- Trading card game for programmers
- Classroom activity about tech pioneers
- Gift for developer friends
- Office decoration / conversation starter

## 📜 License & Attribution

- **Code**: MIT License (your code)
- **Images**: Varies by source
  - Wikipedia images: Check individual licenses (usually CC-BY-SA or Public Domain)
  - Each card shows attribution on footer
  - Metadata stored in `cards/quartetto/wiki_images/metadata/`

## 🤝 Contributing

Want to add more scientists or improve the design?

1. Fork the repo
2. Add scientists to `informatici.txt` and `informatici_cards_analysis.json`
3. Run `python3 generate_quartetto_full.py`
4. Submit a pull request!

## 🐛 Troubleshooting

### "No module named 'img2pdf'"
```bash
pip install img2pdf
```

### "Wikipedia image not found"
Check `cards/quartetto/wiki_images/metadata/*.json` for error details. Some scientists may not have Wikipedia images - placeholders will be created automatically.

### "Font not found"
The renderer will use default font if system fonts aren't available. Install Helvetica/Arial/DejaVu for best results.

### PDF generation fails
Ensure img2pdf and pikepdf are installed:
```bash
pip install img2pdf pikepdf
```

## 📚 Related Documentation

- [QUARTETTO_ROADMAP.md](QUARTETTO_ROADMAP.md) - Original implementation plan
- [Wikipedia API Docs](https://www.mediawiki.org/wiki/API:Main_page)
- [PIL/Pillow Docs](https://pillow.readthedocs.io/)

---

**Made with ❤️ for the NIMITZ project**
