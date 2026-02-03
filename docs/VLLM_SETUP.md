# Configurazione vLLM (o altri endpoint OpenAI-compatible)

NIMITZ supporta l'utilizzo di endpoint personalizzati compatibili con l'API OpenAI, come:
- **vLLM** (servizio locale o remoto)
- **LocalAI**
- **Ollama** (con compatibilità OpenAI)
- **Text Generation WebUI** (con estensione OpenAI)
- Qualsiasi altro servizio che implementa l'API OpenAI

---

## Setup Rapido

### 1. Configurare le variabili d'ambiente

```bash
# URL del tuo endpoint vLLM (obbligatorio)
export VLLM_BASE_URL="http://localhost:8000/v1"

# API key (opzionale - alcuni servizi locali non la richiedono)
export VLLM_API_KEY="dummy"  # o lascia vuoto se non serve
```

### 2. Usare con NIMITZ

```bash
# Modo 1: Auto-detect (se VLLM_BASE_URL è impostato, viene usato automaticamente)
nimitz retrieve discover "baseball italiani" -o players.txt

# Modo 2: Specificare esplicitamente vllm
nimitz retrieve discover "baseball italiani" -o players.txt --llm-provider vllm
```

---

## Configurazione Dettagliata

### Esempio: vLLM locale

```bash
# 1. Avvia il server vLLM (esempio)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000

# 2. Configura NIMITZ
export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_API_KEY="dummy"

# 3. Testa
nimitz retrieve discover "italian painters" --llm-provider vllm
```

### Esempio: vLLM remoto con autenticazione

```bash
export VLLM_BASE_URL="https://your-vllm-server.com/v1"
export VLLM_API_KEY="your-actual-api-key"

nimitz retrieve discover "famous actors" --llm-provider vllm
```

### Esempio: Ollama con compatibilità OpenAI

```bash
# 1. Avvia Ollama con server OpenAI-compatible
ollama serve

# 2. In un altro terminale
export VLLM_BASE_URL="http://localhost:11434/v1"
export VLLM_API_KEY="dummy"

# 3. Usa NIMITZ
nimitz retrieve discover "scientists" --llm-provider vllm
```

---

## Configurazione del Modello

Per cambiare il modello usato, devi modificare il file `src/llm_analyzer.py`:

```python
# Cerca questa sezione:
"vllm": LLMConfig(
    provider="vllm",
    model="openai/vllm",  # <-- Cambia questo se necessario
    api_key=os.environ.get("VLLM_API_KEY", "dummy"),
    base_url=os.environ.get("VLLM_BASE_URL"),
),
```

**Nota:** La maggior parte dei server OpenAI-compatible ignorano il parametro `model` o lo usano internamente. Verifica la documentazione del tuo servizio.

---

## Priorità Auto-Detection

Quando usi `--llm-provider auto`, NIMITZ cerca i provider in questo ordine:

1. **vLLM** (se `VLLM_BASE_URL` è impostato)
2. **Anthropic Claude** (se `ANTHROPIC_API_KEY` è impostato)
3. **Google Gemini** (se `GEMINI_API_KEY` è impostato)
4. **OpenAI GPT** (se `OPENAI_API_KEY` è impostato)

Per forzare l'uso di vLLM anche se hai altri API key configurati:

```bash
nimitz retrieve discover "query" --llm-provider vllm
```

---

## Troubleshooting

### Errore: "VLLM_BASE_URL environment variable not set"

**Soluzione:** Imposta la variabile d'ambiente:
```bash
export VLLM_BASE_URL="http://localhost:8000/v1"
```

### Errore: Connection refused / timeout

**Cause possibili:**
1. Server vLLM non avviato → Verifica che il server sia in esecuzione
2. URL errato → Controlla che l'URL sia corretto (includi `/v1` alla fine)
3. Firewall → Verifica le regole del firewall

**Debug:**
```bash
# Testa l'endpoint manualmente
curl $VLLM_BASE_URL/models

# Se funziona, dovresti vedere la lista dei modelli
```

### Il modello non risponde correttamente

**Possibili soluzioni:**
1. Verifica che il modello supporti generazione di testo (non solo embedding)
2. Aumenta `max_tokens` in `llm_analyzer.py` se le risposte sono troncate
3. Verifica i log del server vLLM per errori

### Risposte lente

vLLM locale può essere lento se:
- Non hai GPU disponibile
- Il modello è molto grande per la tua RAM/VRAM
- Il server è sovraccarico

**Suggerimenti:**
- Usa modelli più piccoli (es. 7B invece di 70B)
- Abilita quantizzazione (es. AWQ, GPTQ)
- Aumenta batch_size in vLLM se hai risorse

---

## Modelli Consigliati per vLLM

### Per entity filtering (compito di classificazione):

**Leggeri e veloci:**
- `meta-llama/Llama-3.2-3B-Instruct` (ottimo bilanciamento)
- `mistralai/Mistral-7B-Instruct-v0.3`
- `google/gemma-2-9b-it`

**Alta qualità:**
- `meta-llama/Llama-3.1-8B-Instruct` (raccomandato)
- `Qwen/Qwen2.5-14B-Instruct`
- `mistralai/Mixtral-8x7B-Instruct-v0.1`

### Requisiti hardware minimi:

| Modello | VRAM | RAM (CPU only) |
|---------|------|----------------|
| 3B      | 6 GB | 8 GB           |
| 7-8B    | 12 GB| 16 GB          |
| 14B     | 24 GB| 32 GB          |
| 70B+    | 80 GB| 128 GB         |

---

## Esempi di Utilizzo Completo

### Workflow completo con vLLM locale:

```bash
# 1. Setup
export VLLM_BASE_URL="http://localhost:8000/v1"

# 2. Discover entities
nimitz retrieve discover "Italian Renaissance painters" \
    -o painters.txt \
    --llm-provider vllm \
    --max-results 30

# 3. Retrieve images (se hai già il file painters.txt)
nimitz retrieve batch painters.txt \
    -o output_images \
    --preset art

# 4. Generate cards
nimitz generate \
    -i output_images \
    -o trading_cards \
    -c "Name,Period,Famous Works,Style"
```

### Confronto tra provider:

```bash
# Test con diversi provider per vedere quale dà risultati migliori

# vLLM locale (gratis, privato)
nimitz retrieve discover "actors" -o test_vllm.txt --llm-provider vllm

# Claude (migliore qualità, a pagamento)
nimitz retrieve discover "actors" -o test_claude.txt --llm-provider anthropic

# Confronta i risultati
diff test_vllm.txt test_claude.txt
```

---

## Configurazione Avanzata

### Custom model path in codice

Se hai bisogno di configurazioni più complesse, modifica `src/llm_analyzer.py`:

```python
# Esempio: modello specifico con parametri custom
"vllm": LLMConfig(
    provider="vllm",
    model="meta-llama/Llama-3.1-8B-Instruct",  # Specifica il modello
    api_key=os.environ.get("VLLM_API_KEY", ""),
    base_url=os.environ.get("VLLM_BASE_URL"),
    max_tokens=4096,  # Aumenta se serve
    temperature=0.1,  # Bassa per classificazione deterministica
),
```

### Multiple endpoints

Se vuoi usare diversi endpoint vLLM, puoi:

```bash
# Setup endpoint 1 (classificazione)
export VLLM_BASE_URL="http://server1:8000/v1"

# Oppure switch dinamico
alias vllm1='export VLLM_BASE_URL="http://server1:8000/v1"'
alias vllm2='export VLLM_BASE_URL="http://server2:8000/v1"'

# Usa
vllm1
nimitz retrieve discover "query1" -o out1.txt --llm-provider vllm

vllm2
nimitz retrieve discover "query2" -o out2.txt --llm-provider vllm
```

---

## FAQ

**Q: Posso usare vLLM per l'analisi delle immagini?**  
A: Sì, se il tuo modello vLLM supporta vision (es. LLaVA, Qwen-VL). Assicurati che il server sia configurato per modelli multimodali.

**Q: vLLM è compatibile con tutti i modelli?**  
A: vLLM supporta la maggior parte dei modelli transformer-based. Vedi [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models.html).

**Q: Posso usare più provider contemporaneamente?**  
A: Sì, specifica `--llm-provider` per ogni comando. L'auto-detection usa solo un provider per volta.

**Q: Come verifico quale provider sta usando NIMITZ?**  
A: Il tool stampa un messaggio all'inizio (es. "🤖 Using LLM to filter results..."). Per debug dettagliato, aggiungi logging nel codice.

---

## Link Utili

- [vLLM Documentation](https://docs.vllm.ai/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [litellm Documentation](https://docs.litellm.ai/)
- [NIMITZ GitHub](https://github.com/your-repo/nimitz) *(aggiorna con il link reale)*

---

**Ultima revisione:** 2024  
**Versione NIMITZ:** 1.0+ (con supporto vLLM)
