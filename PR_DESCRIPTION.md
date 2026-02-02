# Add Image Retrieval and Web Discovery features (Roadmap #6)

## Summary

Implements **Image Retrieval** feature (Roadmap item #6) plus an enhancement with **Web Discovery** capability.

This enables creating card decks from text descriptions without having pre-existing images, and even automatically discovering entities (players, people, etc.) from web searches.

### Features Implemented

#### 1. Image Retrieval
- 🌐 **Multi-source API support**: Unsplash and Pexels with permissive licenses
- 🎯 **CLIP-based intelligent image selection**: Automatically chooses the best image from multiple candidates
- 💾 **Smart caching**: Local cache to avoid redundant downloads
- 📋 **License tracking**: Complete metadata with attribution and license info for every image
- 🖼️ **Placeholder fallback**: Generates placeholder images for failed retrievals
- 📦 **Batch processing**: Supports .txt, .csv, and .json input files
- 🔄 **Full pipeline integration**: Complete integration with CLIP analysis and card generation

#### 2. Web Discovery (Enhancement)
- 🔍 **Brave Search integration**: Automatically discover entities from web searches
- 🤖 **Intelligent entity extraction**: Smart name extraction from search results
- ✨ **Interactive selection**: Review and refine discovered entities
- 📝 **Description templates**: Create rich descriptions using templates (e.g., "{name}, baseball player")
- ⚡ **Auto-mode**: Discover + retrieve + generate cards in one command

### New CLI Commands

```bash
# Check API configuration
nimitz retrieve status

# Retrieve single image
nimitz retrieve single "Golden Gate Bridge at sunset"

# Batch retrieve from file
nimitz retrieve batch players.txt --preset art

# 🆕 Discover entities from web and auto-generate cards
nimitz retrieve discover "Parma Clima Baseball roster" -o players.txt --auto
```

### Example Use Case

Create a baseball card deck without manually typing names:

```bash
# Set up API keys (free tier available for all)
export UNSPLASH_ACCESS_KEY="your-key"
export BRAVE_API_KEY="your-key"

# One command to:
# 1. Search web for team roster
# 2. Extract player names
# 3. Download player images
# 4. Generate trading cards
nimitz retrieve discover "Parma Clima Baseball 2024 roster" \
  -o players.txt \
  --template "{name}, baseball player" \
  --auto \
  --preset art
```

### Files Changed

- ✅ `src/image_retrieval.py` (new, 564 lines) - Image retrieval core
- ✅ `src/web_discovery.py` (new, 379 lines) - Web discovery with Brave Search
- ✅ `src/cli.py` (+640 lines) - CLI integration
- ✅ `requirements.txt` - Added `requests` dependency
- ✅ `README.md` - Complete documentation section
- ✅ `ROADMAP.md` - Feature #6 marked as done + documentation
- ✅ `test_descriptions.txt` - Example input file

### Requirements

**For Image Retrieval** (at least one required):
- `UNSPLASH_ACCESS_KEY` - Free at https://unsplash.com/developers
- `PEXELS_API_KEY` - Free at https://www.pexels.com/api/

**For Web Discovery** (optional):
- `BRAVE_API_KEY` - 2,000 queries/month free at https://brave.com/search/api/

## Test Plan

- [x] Syntax validation - All Python files compile without errors
- [ ] Test `nimitz retrieve status` - Verify API configuration check
- [ ] Test `nimitz retrieve single` - Retrieve and analyze single image
- [ ] Test `nimitz retrieve batch` - Batch processing from .txt file
- [ ] Test `nimitz retrieve discover` - Web discovery with interactive mode
- [ ] Test `nimitz retrieve discover --auto` - End-to-end auto-generation
- [ ] Test caching - Verify cached images are reused
- [ ] Test placeholder fallback - Check placeholder generation on failures
- [ ] Test license metadata - Verify attribution info is preserved
- [ ] Integration test - Full pipeline from discovery to card generation

### Manual Testing Commands

```bash
# Test status check
nimitz retrieve status

# Test single retrieval
nimitz retrieve single "Sunset over ocean" -o ./test_output

# Test batch retrieval
echo -e "Golden Gate Bridge\nEiffel Tower\nMount Fuji" > test.txt
nimitz retrieve batch test.txt -o ./test_batch

# Test discovery (requires BRAVE_API_KEY)
nimitz retrieve discover "Famous landmarks" -o landmarks.txt -n 5
```

https://claude.ai/code/session_01D3zpzNuY4JWS1WTZjzUCXR
