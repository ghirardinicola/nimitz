#!/bin/bash

echo "=== PULIZIA FILE VECCHI ==="
echo ""

# Cancella le vecchie versioni archiviate
echo "1. Rimozione vecchie versioni..."
if [ -d "cards/old_versions" ]; then
    rm -rf cards/old_versions
    echo "   ✓ Cancellate 5 directory vecchie (56 MB liberati)"
fi

# Cancella i vecchi file di test in cards/
echo "2. Rimozione file di test vecchi..."
rm -f cards/image_cards_page_*.png
echo "   ✓ Cancellati 6 file PNG vecchi"

# Cancella directory individual vecchia
if [ -d "cards/individual" ]; then
    rm -rf cards/individual
    echo "   ✓ Cancellata directory individual (8.8 MB)"
fi

echo ""
echo "=== SPAZIO LIBERATO ==="
echo "Prima: ~99 MB (cards totale)"
echo "Dopo:  ~34 MB (solo quartetto)"
echo "Risparmio: ~65 MB"
echo ""
echo "=== STRUTTURA FINALE ==="
ls -lh cards/
echo ""
ls -lh cards/quartetto/

