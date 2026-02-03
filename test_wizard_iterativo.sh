#!/bin/bash

# Test del wizard interattivo e iterativo

set -a
source .env
set +a

echo "=========================================="
echo "Test Wizard Iterativo - Informatici Famosi"
echo "=========================================="
echo ""
echo "Simulo un'interazione iterativa:"
echo "1. Prima iterazione: genera vocabolario base"
echo "2. Feedback: 'aggiungi caratteristiche sul contributo tecnico specifico'"
echo "3. Seconda iterazione: rigenera con feedback"
echo "4. Accetta suggerimenti"
echo ""

# Simulazione input:
# 1. "informatici famosi" - contesto iniziale
# 2. "2" - scelta iterazione (migliora)
# 3. "aggiungi caratteristiche sul contributo tecnico specifico, come linguaggi creati o algoritmi inventati" - feedback
# 4. "1" - accetta suggerimenti raffinati
# 5. "done" - finisci wizard
# 6. "vocabolario_iterativo.json" - salva file

cat <<'EOF' | nimitz wizard
informatici famosi
2
aggiungi caratteristiche sul contributo tecnico specifico, come linguaggi creati o algoritmi inventati
1
done
vocabolario_iterativo.json
EOF

echo ""
echo "=========================================="
echo "Test completato!"
echo "Verifica il file: vocabolario_iterativo.json"
echo "=========================================="
