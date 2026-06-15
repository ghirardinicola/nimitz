#!/bin/bash

echo "=== STRUTTURA DIRECTORY CARDS ==="
echo ""
du -sh cards/*
echo ""
echo "=== CONTENUTO QUARTETTO ==="
ls -lh cards/quartetto/ | grep -v "^total"
echo ""
echo "=== PDF FINALE ==="
ls -lh cards/quartetto/poker/*.pdf 2>/dev/null || echo "PDF non trovato"
echo ""
echo "=== IMMAGINI WIKIPEDIA ==="
echo "Totale immagini: $(ls cards/quartetto/wiki_images/*.{png,jpg,jpeg,JPG} 2>/dev/null | wc -l)"
echo ""
echo "=== VECCHIE VERSIONI (da cancellare) ==="
ls -d cards/old_versions/*/ 2>/dev/null | wc -l
echo "directory archiviate"
