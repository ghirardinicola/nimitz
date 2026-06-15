# 🎯 NIMITZ Quartetto Cards - Implementation Roadmap

## Project Goal
Transform NIMITZ from Pokemon-style cards to Top Trumps/Quartetto style cards with:
- Wikipedia images (authentic computer scientist photos)
- English language
- Print-ready poker card format (63×88mm)
- All statistics visible with descriptions
- Top skill highlighted

---

## ✅ DECISIONS CONFIRMED

1. **Missing Wikipedia images** → Styled placeholder with scientist name
2. **Card back** → Unified design with NIMITZ logo
3. **Language** → English (translate Italian vocabulary)
4. **Custom images** → No custom images needed
5. **Print layout** → A4 with 3×3 grid (9 cards/page) ✓
6. **Numbering** → Simple sequential 1-32

---

## 📋 IMPLEMENTATION PHASES

### **PHASE 1: Foundation (Sprint 1 - ~3 hours)**

#### Task 1.1: Translate Vocabulary to English
**File**: `vocabolario_informatici_game_en.json` (NEW)

**Translations**:
- `influenza_sul_mercato` → `market_influence` → "Market Influence"
- `uso_di_tecnologie_avanzate` → `advanced_technologies` → "Advanced Tech"
- `riconoscimento_professionale` → `professional_recognition` → "Recognition"
- `lunghezza_della_barba` → `beard_length` → "Beard Length"
- `openess` → `openness` → "Openness"

**Status**: [ ] Not Started

---

#### Task 1.2: Create Wikipedia Image Retriever
**File**: `src/wikimedia_retrieval.py` (NEW ~250 lines)

**Key Functions**:
```python
class WikimediaRetriever:
    def search_wikipedia_page(self, name: str) -> Optional[str]
    def get_page_image_url(self, page_title: str) -> Optional[Dict]
    def download_image(self, url: str, output_path: str) -> bool
    def create_placeholder(self, name: str, output_path: str) -> bool
    def get_scientist_image(self, name: str, output_dir: str) -> Dict
```

**Placeholder Design**:
- Background: #E5E5E5 (light gray)
- Border: 2px solid #999999
- Text: Scientist name + "[Photo unavailable]"
- Size: 300×300px minimum

**Status**: [ ] Not Started

---

#### Task 1.3: Test Wikipedia Retrieval
**Test with 5 scientists**:
1. Alan Turing (should find image)
2. Ada Lovelace (should find image)
3. Grace Hopper (should find image)
4. Unknown Scientist (should create placeholder)
5. One random from list

**Output**: `./test_wiki_images/`

**Status**: [ ] Not Started

**Checkpoint**: Review 5 sample images before proceeding

---

### **PHASE 2: Data Enrichment (Sprint 2 - ~1.5 hours)**

#### Task 2.1: Add JSON Enrichment Function
**File**: Modify `create_deck_step3_enriched.py` (lines ~467-478)

**New Function**:
```python
def enrich_card_with_descriptions(card: Dict, vocab: Dict) -> Dict:
    # Map score (0-100) to description level (0-4)
    # Calculate rank for each characteristic
    # Identify top skill
    # Add display names
    # Return enriched structure
```

**Enhanced JSON Structure**:
```json
{
  "name": "Ada Lovelace",
  "image": "./path/to/image.jpg",
  "image_source": "wikipedia",
  "attribution": "Wikipedia CC-BY-SA",
  "scores": {
    "market_influence": {
      "value": 74,
      "level": 3,
      "description": "developer with great impact...",
      "rank": 2,
      "display_name": "Market Influence"
    }
  },
  "top_skill": {
    "key": "advanced_technologies",
    "display_name": "Advanced Tech",
    "value": 75,
    "description": "developer with advanced skills..."
  }
}
```

**Status**: [ ] Not Started

---

#### Task 2.2: Generate Enriched JSON
**Input**: `informatici_cards_analysis.json`
**Output**: `informatici_cards_analysis_enriched.json`

**Status**: [ ] Not Started

**Checkpoint**: Validate enriched JSON structure

---

### **PHASE 3: Card Design (Sprint 3 - ~3 hours)**

#### Task 3.1: Create Quartetto Card Renderer
**File**: `src/card_renderer_quartetto.py` (NEW ~400 lines)

**Card Layout (Poker Size: 744 × 1039 pixels)**:
```
┌─────────────────────────────────────┐ 0px
│ ALAN TURING                  #12/32 │ Header: 60px (navy)
├─────────────────────────────────────┤ 60px
│         [Photo 300×300px]           │ Photo section: 340px
├─────────────────────────────────────┤ 400px
│ 🏆 TOP SKILL:                       │ Highlight: 80px (gold)
│ Advanced Tech                    75 │
│ "developer with advanced skills..." │
├─────────────────────────────────────┤ 480px
│ ALL STATISTICS:                     │ Stats: 450px
│ Market Influence               74   │
│ Advanced Tech                  75 ★ │
│ Recognition                    74   │
│ Beard Length                   60   │
│ Openness                       74   │
├─────────────────────────────────────┤ 930px
│ Wikipedia CC-BY-SA        Card 12/32│ Footer: 40px
└─────────────────────────────────────┘ 1039px
```

**Colors**:
- Header BG: #1A2332 (navy)
- Top skill BG: #FFD700 (gold)
- Stats BG: #F8F8F8 (light gray)
- Main BG: #FFFFFF (white)
- Borders: #CCCCCC
- Text: #333333

**Key Methods**:
```python
class QuartettoCardRenderer:
    def __init__(self, card_size='poker')
    def render_card(self, card_data: Dict) -> Image
    def render_card_back(self) -> Image
    def save_card(self, image: Image, output_path: str)
    def render_grid_page(self, cards: List[Image]) -> Image
    def add_cut_marks(self, page: Image) -> Image
    def export_pdf(self, cards: List[Dict], output_path: str)
```

**Status**: [ ] Not Started

---

#### Task 3.2: Design Card Back
**Unified back design**:
- Navy gradient background (#1A2332 to #2A3342)
- "NIMITZ" large text (white)
- "Computer Scientist Cards" subtitle
- Anchor symbol (⚓) in gold
- Thin gold border

**Status**: [ ] Not Started

---

#### Task 3.3: Test Card Rendering
**Render 3 test cards**:
1. High scores (all 80+)
2. Medium scores (all 50-70)
3. Low scores (all 20-40)

**Validate**:
- [ ] Text doesn't overflow
- [ ] Images display correctly
- [ ] Top skill highlighted
- [ ] All 5 stats visible
- [ ] Descriptions readable
- [ ] Card dimensions exact (744×1039)

**Status**: [ ] Not Started

**Checkpoint**: Review 3 sample cards for design approval

---

### **PHASE 4: Full Generation (Sprint 4 - ~1.5 hours)**

#### Task 4.1: Download All Wikipedia Images
**Input**: `informatici.txt` (32 scientists)
**Output**: `informatici_cards_wiki/` (32 images or placeholders)

**Status**: [ ] Not Started

---

#### Task 4.2: Generate All Card Formats
**Outputs**:
```
informatici_cards_quartetto/
├── poker/
│   ├── fronts/    (32 × 744×1039px)
│   └── backs/     (1 × 744×1039px)
├── display/       (32 × 600×850px)
├── grids/         (8 × A4 pages with cut marks)
└── quartetto_deck_print.pdf (8 pages)
```

**Status**: [ ] Not Started

---

#### Task 4.3: Quality Review
**Check**:
- [ ] All 32 scientists have cards
- [ ] Images are appropriate (no stock photos)
- [ ] Text is readable at poker size
- [ ] Descriptions match score levels
- [ ] Top skills correctly identified
- [ ] Attribution text present
- [ ] PDF cuts correctly
- [ ] Print dimensions accurate (63×88mm @ 300 DPI)

**Status**: [ ] Not Started

---

#### Task 4.4: Create Helper Scripts
**Files**:
1. `generate_quartetto_cards.py` - Standalone card generator
2. `test_quartetto_cards.py` - Test suite
3. Update `run_workflow.sh` - Add Wikipedia + Quartetto steps

**Status**: [ ] Not Started

---

## 📦 DELIVERABLES

### Immediate Outputs:
- [x] This roadmap file
- [ ] English vocabulary JSON
- [ ] Wikipedia image retriever
- [ ] 32 Wikipedia images (or placeholders)
- [ ] Enriched JSON with descriptions
- [ ] Quartetto card renderer
- [ ] 32 individual cards (poker size)
- [ ] 32 display cards (screen size)
- [ ] 4 grid pages (9 cards each)
- [ ] Print-ready PDF (8 pages)
- [ ] Card back design
- [ ] Test suite
- [ ] Helper scripts

---

## 🚦 CURRENT STATUS

**Last Updated**: 2025-02-05 16:45

**Current Phase**: Phase 1 - Foundation
**Current Task**: Task 1.1 - Translate Vocabulary
**Progress**: 0% (roadmap created)

---

## 📊 COMPLETION CHECKLIST

### Phase 1: Foundation
- [ ] Task 1.1: Vocabulary translation complete
- [ ] Task 1.2: Wikipedia retriever implemented
- [ ] Task 1.3: 5 test images validated
- [ ] Checkpoint: Sample images approved

### Phase 2: Data Enrichment
- [ ] Task 2.1: Enrichment function added
- [ ] Task 2.2: Enriched JSON generated
- [ ] Checkpoint: JSON structure validated

### Phase 3: Card Design
- [ ] Task 3.1: Card renderer implemented
- [ ] Task 3.2: Card back designed
- [ ] Task 3.3: 3 test cards rendered
- [ ] Checkpoint: Card design approved

### Phase 4: Full Generation
- [ ] Task 4.1: All 32 Wikipedia images downloaded
- [ ] Task 4.2: All card formats generated
- [ ] Task 4.3: Quality review passed
- [ ] Task 4.4: Helper scripts created
- [ ] Final: Complete deck ready!

---

## 🔧 TECHNICAL SPECIFICATIONS

### Card Dimensions:
- **Poker (Print)**: 744 × 1039 pixels (63mm × 88mm @ 300 DPI)
- **Display (Screen)**: 600 × 850 pixels

### Print Layout:
- **Page Size**: A4 (2480 × 3508 pixels @ 300 DPI)
- **Cards per Page**: 9 (3×3 grid)
- **Total Pages**: 4 fronts + 4 backs = 8 pages
- **Margins**: 30px page margin, 20px between cards
- **Cut Marks**: 5mm corner marks

### Dependencies:
```bash
pip install pillow reportlab wikipedia-api requests
```

---

## ⚠️ KNOWN ISSUES & MITIGATIONS

| Issue | Mitigation | Status |
|-------|-----------|--------|
| Wikipedia image not found | Styled placeholder | ✓ Planned |
| Image too low resolution | Fallback or placeholder | ✓ Planned |
| Text overflow on card | Truncate + smaller font | ✓ Planned |
| Long scientist names | 2 lines, reduced font | ✓ Planned |
| API rate limiting | Add delays | ✓ Planned |

---

## 📝 NOTES

- Original Pokemon-style cards in `informatici_cards_visual/` (deprecated)
- Original Pexels images in `informatici_cards/` (will be replaced)
- Original Italian vocabulary in `vocabolario_informatici_game.json`
- Current analysis JSON in `informatici_cards_analysis.json`
- CLIP model used for scoring (keep existing)
- LLM + CLIP hybrid scoring (70% LLM, 30% CLIP)

---

## 🎓 REFERENCES

- Top Trumps official site: https://www.toptrumps.com
- Wikipedia API docs: https://www.mediawiki.org/wiki/API:Main_page
- Wikimedia Commons: https://commons.wikimedia.org
- Poker card standard: 63mm × 88mm (2.48" × 3.46")
- Print DPI: 300 (industry standard)

---

**End of Roadmap**
