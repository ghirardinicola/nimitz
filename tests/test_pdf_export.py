"""
Tests for NIMITZ PDF Export Module
"""

import pytest
import tempfile
from pathlib import Path
from pdf_export import (
    check_pdf_support,
    get_card_color,
    export_cards_to_pdf,
    export_cards_to_png,
    PIL_AVAILABLE,
    REPORTLAB_AVAILABLE,
)
from gaming import RarityTier, RARITY_COLORS


class TestPdfSupport:
    """Tests for PDF support checking"""

    def test_check_pdf_support_returns_bool(self):
        """check_pdf_support should return boolean"""
        result = check_pdf_support()

        assert isinstance(result, bool)

    def test_pdf_support_depends_on_libraries(self):
        """PDF support depends on both reportlab and PIL"""
        result = check_pdf_support()

        # Should be True only if both are available
        assert result == (REPORTLAB_AVAILABLE and PIL_AVAILABLE)


class TestCardColor:
    """Tests for card color getting"""

    def test_get_card_color_valid_tiers(self):
        """Valid tiers should return their colors"""
        for tier in RarityTier:
            color = get_card_color(tier.value)

            assert color.startswith("#")
            assert color == RARITY_COLORS[tier]

    def test_get_card_color_invalid_tier(self):
        """Invalid tier should return default color"""
        color = get_card_color("nonexistent_tier")

        assert color == "#333333"

    def test_get_card_color_common(self):
        """Common tier should return gray"""
        color = get_card_color("common")

        assert color == "#9E9E9E"

    def test_get_card_color_legendary(self):
        """Legendary tier should return gold/orange"""
        color = get_card_color("legendary")

        assert color == "#FF9800"


class TestExportPdf:
    """Tests for PDF export functionality"""

    @pytest.mark.skipif(
        not (REPORTLAB_AVAILABLE and PIL_AVAILABLE),
        reason="PDF export requires reportlab and PIL"
    )
    def test_export_cards_to_pdf_basic(self, sample_cards_collection):
        """Test basic PDF export"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            filepath = f.name

        try:
            result = export_cards_to_pdf(
                sample_cards_collection,
                filepath,
                page_size="A4"
            )

            assert result.exists()
            assert result.suffix == '.pdf'
            # Check file has content
            assert result.stat().st_size > 0
        finally:
            Path(filepath).unlink(missing_ok=True)

    @pytest.mark.skipif(
        not (REPORTLAB_AVAILABLE and PIL_AVAILABLE),
        reason="PDF export requires reportlab and PIL"
    )
    def test_export_cards_to_pdf_letter_size(self, sample_cards_collection):
        """Test PDF export with Letter page size"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            filepath = f.name

        try:
            result = export_cards_to_pdf(
                sample_cards_collection,
                filepath,
                page_size="Letter"
            )

            assert result.exists()
        finally:
            Path(filepath).unlink(missing_ok=True)

    @pytest.mark.skipif(
        not (REPORTLAB_AVAILABLE and PIL_AVAILABLE),
        reason="PDF export requires reportlab and PIL"
    )
    def test_export_cards_to_pdf_adds_extension(self, sample_card):
        """Export should add .pdf extension if missing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "cards_no_extension"

            result = export_cards_to_pdf([sample_card], str(filepath))

            assert result.suffix == '.pdf'
            assert result.exists()

    def test_export_cards_to_pdf_empty_list(self):
        """Exporting empty list should raise ValueError"""
        with pytest.raises(ValueError, match="No cards"):
            export_cards_to_pdf([], "output.pdf")

    @pytest.mark.skipif(
        REPORTLAB_AVAILABLE and PIL_AVAILABLE,
        reason="Test only runs when dependencies are missing"
    )
    def test_export_cards_to_pdf_missing_dependencies(self, sample_card):
        """Export should raise ImportError when dependencies missing"""
        with pytest.raises(ImportError):
            export_cards_to_pdf([sample_card], "output.pdf")


class TestExportPng:
    """Tests for PNG export functionality"""

    @pytest.mark.skipif(
        not PIL_AVAILABLE,
        reason="PNG export requires PIL"
    )
    def test_export_cards_to_png_basic(self, sample_cards_collection):
        """Test basic PNG export"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_cards_to_png(sample_cards_collection, tmpdir)

            assert len(result) == len(sample_cards_collection)
            for path in result:
                assert path.exists()
                assert path.suffix == '.png'

    @pytest.mark.skipif(
        not PIL_AVAILABLE,
        reason="PNG export requires PIL"
    )
    def test_export_cards_to_png_creates_directory(self, sample_card):
        """Export should create output directory if it doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "new_subdir" / "cards"

            result = export_cards_to_png([sample_card], str(output_dir))

            assert output_dir.exists()
            assert len(result) == 1

    @pytest.mark.skipif(
        not PIL_AVAILABLE,
        reason="PNG export requires PIL"
    )
    def test_export_cards_to_png_sanitizes_filename(self, sample_card):
        """Filenames should be sanitized for special characters"""
        # Card with special characters in name
        card = sample_card.copy()
        card['image_name'] = 'test/photo:special*chars?.jpg'

        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_cards_to_png([card], tmpdir)

            assert len(result) == 1
            # Filename should not contain special chars
            filename = result[0].name
            assert '/' not in filename
            assert ':' not in filename
            assert '*' not in filename
            assert '?' not in filename


class TestExportIntegration:
    """Integration tests for export functionality"""

    @pytest.mark.skipif(
        not (REPORTLAB_AVAILABLE and PIL_AVAILABLE),
        reason="Integration tests require all dependencies"
    )
    def test_export_pdf_with_gaming_stats(self, sample_cards_collection):
        """PDF export should include gaming stats"""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            filepath = f.name

        try:
            result = export_cards_to_pdf(
                sample_cards_collection,
                filepath,
                include_stats=True
            )

            # Just verify it completes without error
            assert result.exists()
        finally:
            Path(filepath).unlink(missing_ok=True)

    @pytest.mark.skipif(
        not PIL_AVAILABLE,
        reason="PNG export requires PIL"
    )
    def test_export_png_file_sizes(self, sample_cards_collection):
        """PNG files should have reasonable sizes"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = export_cards_to_png(sample_cards_collection, tmpdir)

            for path in result:
                size = path.stat().st_size
                # Should be at least 1KB and less than 1MB
                assert size > 1024
                assert size < 1024 * 1024
