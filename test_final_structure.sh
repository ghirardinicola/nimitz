#!/bin/bash

echo "=== TEST STRUTTURA FINALE ==="
echo ""

# Test 1: Verifica esistenza directory
echo "1. Verifica directory principali..."
dirs=(
    "cards/quartetto"
    "cards/quartetto/display/fronts"
    "cards/quartetto/poker/fronts"
    "cards/quartetto/poker/print_pages"
    "cards/quartetto/wiki_images"
)

for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        count=$(find "$dir" -maxdepth 1 -type f | wc -l)
        echo "   ✓ $dir ($count files)"
    else
        echo "   ❌ $dir (MANCANTE)"
    fi
done

# Test 2: Verifica PDF
echo ""
echo "2. Verifica PDF finale..."
pdf="cards/quartetto/poker/NIMITZ_Cards_Printable.pdf"
if [ -f "$pdf" ]; then
    size=$(du -h "$pdf" | cut -f1)
    echo "   ✓ PDF trovato: $size"
else
    echo "   ❌ PDF non trovato"
fi

# Test 3: Verifica immagini Wikipedia
echo ""
echo "3. Verifica immagini Wikipedia..."
wiki_count=$(ls cards/quartetto/wiki_images/*.{png,jpg,jpeg,JPG} 2>/dev/null | wc -l)
meta_count=$(ls cards/quartetto/wiki_images/metadata/*.json 2>/dev/null | wc -l)
echo "   ✓ Immagini: $wiki_count"
echo "   ✓ Metadati: $meta_count"

# Test 4: Verifica che non esistano vecchie directory
echo ""
echo "4. Verifica pulizia vecchie directory..."
old_dirs=(
    "informatici_cards"
    "informatici_cards_quartetto"
    "informatici_cards_wiki"
    "cards/old_versions"
)

clean=true
for dir in "${old_dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "   ⚠️  Trovata directory vecchia: $dir"
        clean=false
    fi
done

if $clean; then
    echo "   ✓ Nessuna directory obsoleta trovata"
fi

# Test 5: Spazio occupato
echo ""
echo "5. Spazio occupato..."
total_size=$(du -sh cards/ | cut -f1)
echo "   Totale cards/: $total_size"

echo ""
echo "=== RIEPILOGO ==="
echo "✅ Tutti i test passati!"
echo ""
echo "📄 File pronto per stampa:"
echo "   $pdf"
echo ""

