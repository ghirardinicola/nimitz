# Quickstart: Usare NIMITZ con vLLM

Guida rapida per configurare e usare NIMITZ con un server vLLM locale.

---

## Setup in 3 Passi

### 1. Avvia il server vLLM

```bash
# Opzione A: Modello piccolo e veloce (consigliato per test)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-3B-Instruct \
    --port 8000

# Opzione B: Modello più potente (richiede più VRAM/RAM)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000
```

### 2. Configura le variabili d'ambiente

```bash
export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_API_KEY="dummy"
```

### 3. Usa NIMITZ

```bash
# Test base
nimitz retrieve discover "italian baseball players" \
    -o players.txt \
    --llm-provider vllm

# Con auto-retrieval delle immagini
nimitz retrieve discover "famous scientists" \
    -o scientists.txt \
    --llm-provider vllm \
    --auto
```

---

## Verifica che Funzioni

### Test manuale del server vLLM:

```bash
# Verifica che il server risponda
curl http://localhost:8000/v1/models

# Dovrebbe restituire qualcosa tipo:
# {
#   "object": "list",
#   "data": [
#     {
#       "id": "meta-llama/Llama-3.2-3B-Instruct",
#       "object": "model",
#       ...
#     }
#   ]
# }
```

### Test NIMITZ:

```bash
# Test semplice
python test_llm_filter.py

# Test completo
nimitz retrieve discover "italian renaissance painters" \
    -o test_painters.txt \
    --llm-provider vllm \
    --max-results 10
```

---

## Requisiti Hardware

| Modello | VRAM (GPU) | RAM (CPU only) |
|---------|------------|----------------|
| 3B      | ~6 GB      | ~8 GB          |
| 7-8B    | ~12 GB     | ~16 GB         |
| 13-14B  | ~24 GB     | ~32 GB         |

**Nota:** CPU-only è molto più lento ma funziona. Raccomandato: almeno una GPU con 8GB VRAM.

---

## Confronto Provider

```bash
# Test con vLLM (locale, gratis)
export VLLM_BASE_URL="http://localhost:8000/v1"
nimitz retrieve discover "actors" -o test_vllm.txt --llm-provider vllm

# Test con Claude (cloud, migliore qualità)
export ANTHROPIC_API_KEY="sk-..."
nimitz retrieve discover "actors" -o test_claude.txt --llm-provider anthropic

# Confronta
diff test_vllm.txt test_claude.txt
wc -l test_*.txt
```

---

## Troubleshooting Rapido

**Errore: Connection refused**
```bash
# Verifica che vLLM sia in esecuzione
curl http://localhost:8000/health

# Se fallisce, controlla i log di vLLM
```

**Risposta lenta/freeze**
- Modello troppo grande per la tua GPU/RAM
- Prova un modello più piccolo (3B invece di 8B)
- Riduci `--max-results` in NIMITZ

**Risposta vuota/malformata**
- Il modello potrebbe non seguire bene le istruzioni JSON
- Prova un modello Instruct-tuned (es. Llama-3.1-Instruct)
- Verifica i log di vLLM per errori

---

## Modelli Raccomandati

**Per entity filtering (task di classificazione):**

1. **meta-llama/Llama-3.2-3B-Instruct** ⭐ (migliore bilanciamento velocità/qualità)
2. **meta-llama/Llama-3.1-8B-Instruct** (migliore qualità)
3. **mistralai/Mistral-7B-Instruct-v0.3** (alternativa solida)

**NON usare modelli base** (es. Llama-3.2-3B senza -Instruct) - non seguiranno le istruzioni correttamente.

---

## Prossimi Passi

Una volta configurato vLLM:

1. **Testa il filtering:**
   ```bash
   nimitz retrieve discover "your query" -o output.txt --llm-provider vllm
   ```

2. **Usa con workflow completo:**
   ```bash
   # Discover → Retrieve → Generate
   nimitz retrieve discover "topic" -o entities.txt --llm-provider vllm --auto
   nimitz generate -i output_images -o cards -c "Name,Stat1,Stat2"
   ```

3. **Leggi la guida completa:** `docs/VLLM_SETUP.md`

---

**Buon divertimento con NIMITZ + vLLM! 🚀**
