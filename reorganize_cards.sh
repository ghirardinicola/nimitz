#!/bin/bash

echo "=== RIORGANIZZAZIONE CARDS ==="
echo ""

# Crea struttura dentro cards/
mkdir -p cards/quartetto/{display,poker,wiki_images}
mkdir -p cards/old_versions

# Sposta contenuti quartetto
echo "1. Spostamento contenuti quartetto..."
if [ -d "informatici_cards_quartetto" ]; then
    mv informatici_cards_quartetto/* cards/quartetto/ 2>/dev/null || true
    echo "   ✓ Contenuti quartetto spostati"
fi

# Sposta immagini Wikipedia
echo "2. Spostamento immagini Wikipedia..."
if [ -d "informatici_cards_wiki" ]; then
    mv informatici_cards_wiki/* cards/quartetto/wiki_images/ 2>/dev/null || true
    echo "   ✓ Immagini Wikipedia spostate"
fi

# Sposta vecchie versioni
echo "3. Archiviazione vecchie versioni..."
for dir in informatici_cards informatici_cards_visual baseball_cards_final baseball_cards_output test_wiki_images; do
    if [ -d "$dir" ]; then
        mv "$dir" cards/old_versions/ 2>/dev/null || true
        echo "   ✓ Archiviato: $dir"
    fi
done

# Rimuovi directory vuote
echo "4. Pulizia directory vuote..."
rmdir informatici_cards_quartetto informatici_cards_wiki 2>/dev/null || true

echo ""
echo "=== STRUTTURA FINALE ==="
tree -L 2 cards/ -d 2>/dev/null || find cards/ -type d -maxdepth 2 | sort

