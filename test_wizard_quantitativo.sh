#!/bin/bash

# Test del wizard con vocabolario QUANTITATIVO

set -a
source .env
set +a

echo "=========================================="
echo "Test Wizard QUANTITATIVO - Informatici"
echo "=========================================="
echo ""
echo "Genero vocabolario con caratteristiche quantificabili"
echo "per creare carte da gioco stile Top Trumps"
echo ""

# Input:
# 1. "informatici famosi" - contesto
# 2. "1" - accetta prima generazione (ora quantitativa)
# 3. "done" - finisci
# 4. "vocabolario_quantitativo.json" - salva

cat <<'EOF' | nimitz wizard
informatici famosi
1
done
vocabolario_quantitativo.json
EOF

echo ""
echo "=========================================="
echo "Verifica il vocabolario generato:"
cat vocabolario_quantitativo.json | python -m json.tool | head -60
echo "=========================================="
