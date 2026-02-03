# 🎉 Configurazione Completata: NIMITZ + vLLM Leitha

## Riepilogo Setup

✅ **Server vLLM:** `https://agent-codeai.leitha.servizi.gr-u.it/v1`  
✅ **Modello:** `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8`  
✅ **API Key:** `anything`  
✅ **Status:** Server attivo e raggiungibile

---

## 🚀 Uso Rapido

### Opzione 1: Script di Setup Automatico (CONSIGLIATO)

```bash
cd /Users/nic/prj/nimitz
source ./setup_vllm_leitha.sh
```

Lo script:
- ✅ Configura automaticamente le variabili d'ambiente
- ✅ Testa la connessione al server
- ✅ Ti chiede se vuoi salvare la configurazione in `~/.bashrc` o `~/.zshrc`

### Opzione 2: Configurazione Manuale

```bash
# Aggiungi queste righe a ~/.bashrc o ~/.zshrc
export VLLM_BASE_URL="https://agent-codeai.leitha.servizi.gr-u.it/v1"
export VLLM_API_KEY="anything"
export VLLM_MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"

# Ricarica il profilo
source ~/.bashrc  # o ~/.zshrc se usi zsh
```

### Opzione 3: Uso Temporaneo (Solo Sessione Corrente)

```bash
export VLLM_BASE_URL="https://agent-codeai.leitha.servizi.gr-u.it/v1"
export VLLM_API_KEY="anything"
export VLLM_MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
```

---

## 📝 Esempi di Utilizzo

### 1. Discovery Base con vLLM

```bash
nimitz retrieve discover "italian baseball players" \
    -o players.txt \
    --llm-provider vllm
```

**Cosa succede:**
1. 🌐 Cerca sul web "italian baseball players"
2. 🤖 Usa il server vLLM Leitha per filtrare i nomi reali
3. 💬 Ti chiede conferma interattiva
4. 💾 Salva i risultati in `players.txt`

### 2. Discovery con Auto-Detection

Se hai configurato `VLLM_BASE_URL`, vLLM viene usato automaticamente:

```bash
nimitz retrieve discover "famous scientists" -o scientists.txt
```

(Non serve specificare `--llm-provider vllm`, viene rilevato automaticamente!)

### 3. Discovery + Auto-Retrieve Immagini

```bash
nimitz retrieve discover "renaissance painters" \
    -o painters.txt \
    --llm-provider vllm \
    --auto \
    --preset art
```

**Workflow completo:**
1. 🔍 Discover → filtra con vLLM
2. 📸 Scarica automaticamente immagini da Unsplash
3. 💾 Salva tutto in cartelle organizzate

### 4. Workflow Completo: Discover → Generate Cards

```bash
# Step 1: Discover entities
nimitz retrieve discover "italian football players" \
    -o players.txt \
    --llm-provider vllm \
    --auto

# Step 2: Generate trading cards
nimitz generate \
    -i nimitz_output \
    -o trading_cards \
    -c "Name,Position,Goals,Assists,Team"
```

---

## 🧪 Test della Configurazione

### Test 1: Verifica Variabili d'Ambiente

```bash
echo "VLLM_BASE_URL: $VLLM_BASE_URL"
echo "VLLM_API_KEY: $VLLM_API_KEY"
echo "VLLM_MODEL: $VLLM_MODEL"
```

**Output atteso:**
```
VLLM_BASE_URL: https://agent-codeai.leitha.servizi.gr-u.it/v1
VLLM_API_KEY: anything
VLLM_MODEL: Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
```

### Test 2: Verifica Connessione Server

```bash
curl -s "https://agent-codeai.leitha.servizi.gr-u.it/v1/models" \
    -H "Authorization: Bearer anything" | jq .
```

**Output atteso:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
      "object": "model",
      ...
    }
  ]
}
```

### Test 3: Test NIMITZ Help

```bash
nimitz retrieve discover --help | grep vllm
```

**Output atteso:**
```
--llm-provider {auto,anthropic,gemini,openai,vllm}
```

### Test 4: Test Completo (Richiede Server Attivo)

```bash
# Test con dati reali
nimitz retrieve discover "test query" \
    -o test_output.txt \
    --llm-provider vllm \
    --max-results 5
```

---

## 🔧 Troubleshooting

### Problema: "VLLM_BASE_URL environment variable not set"

**Soluzione:**
```bash
# Verifica che le variabili siano impostate
echo $VLLM_BASE_URL

# Se vuoto, impostale di nuovo
export VLLM_BASE_URL="https://agent-codeai.leitha.servizi.gr-u.it/v1"
export VLLM_API_KEY="anything"
export VLLM_MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
```

### Problema: "Connection refused" o timeout

**Cause possibili:**
1. Server vLLM offline
2. Problemi di rete/VPN
3. Firewall che blocca la connessione

**Debug:**
```bash
# Test connessione base
curl -v "https://agent-codeai.leitha.servizi.gr-u.it/v1/models" \
    -H "Authorization: Bearer anything"

# Se fallisce, verifica:
# - Sei connesso alla VPN? (se necessaria)
# - Il server è attivo?
# - Hai accesso a internet?
```

### Problema: Risposte lente o incomplete

Il modello Qwen3-Coder-30B è grande e potente, ma potrebbe essere più lento di GPT/Claude.

**Tips:**
- Riduci `--max-results` per query più veloci
- Il primo run è più lento (cold start del modello)
- Successive query sono più veloci (cache)

### Problema: "LLM filtering failed"

Se vedi questo warning, NIMITZ fallback alla lista originale (senza filtering).

**Possibili cause:**
- Risposta malformata dal modello
- Timeout della richiesta
- Errore di parsing JSON

**Debug:** Controlla i log completi per vedere l'errore esatto.

---

## 📊 Confronto Provider

### vLLM Leitha vs Cloud APIs

| Feature | vLLM Leitha | Claude | Gemini | GPT-4 |
|---------|-------------|--------|--------|-------|
| **Costo** | Gratis (interno) | $$$$ | $$$ | $$$$ |
| **Privacy** | ✅ Interno | ⚠️ Cloud | ⚠️ Cloud | ⚠️ Cloud |
| **Velocità** | 🟡 Media | 🟢 Veloce | 🟢 Veloce | 🟡 Media |
| **Qualità** | 🟢 Buona | 🟢 Ottima | 🟢 Ottima | 🟢 Ottima |
| **Uptime** | 🟡 Dipende | 🟢 99.9% | 🟢 99.9% | 🟢 99.9% |

**Quando usare vLLM Leitha:**
- ✅ Vuoi tenere i dati interni
- ✅ Non vuoi costi API
- ✅ Il server è sempre disponibile
- ✅ La qualità è sufficiente per il task

**Quando usare Cloud APIs:**
- ✅ Massima qualità richiesta
- ✅ Uptime critico
- ✅ vLLM non disponibile

---

## 🎯 Priorità Auto-Detection

Quando usi `--llm-provider auto`, NIMITZ cerca in questo ordine:

1. **vLLM** (se `VLLM_BASE_URL` impostato) ← HAI PRIORITÀ!
2. **Claude** (se `ANTHROPIC_API_KEY` impostato)
3. **Gemini** (se `GEMINI_API_KEY` impostato)
4. **OpenAI** (se `OPENAI_API_KEY` impostato)

Quindi con la tua configurazione, **vLLM Leitha viene sempre usato** di default!

Per forzare un altro provider:
```bash
# Usa Claude anche se vLLM è configurato
nimitz retrieve discover "query" --llm-provider anthropic
```

---

## 📚 File di Riferimento

- **Setup automatico:** `setup_vllm_leitha.sh`
- **Test connessione:** `test_vllm_leitha.py`
- **Guida completa vLLM:** `docs/VLLM_SETUP.md`
- **Quickstart generale:** `QUICKSTART_VLLM.md`
- **Note implementazione:** `IMPLEMENTATION_NOTES.md`

---

## ✅ Checklist Setup Completo

- [x] Variabili d'ambiente configurate
- [x] Server vLLM raggiungibile e testato
- [x] Modello corretto (Qwen3-Coder-30B-A3B-Instruct-FP8)
- [x] NIMITZ aggiornato con supporto vLLM
- [x] CLI mostra opzione "vllm"
- [x] Auto-detection funzionante
- [ ] Test con query reale ← **PROSSIMO PASSO!**

---

## 🚀 Prossimo Passo: Test Reale

Prova subito con una query vera:

```bash
# Setup (se non già fatto)
source setup_vllm_leitha.sh

# Test!
nimitz retrieve discover "giocatori baseball italiani" \
    -o test_baseball.txt \
    --llm-provider vllm \
    --max-results 10
```

Dovresti vedere:
1. 🌐 Ricerca web
2. 🤖 "Using LLM to filter results..."
3. 📋 Lista nomi filtrati
4. 💬 Richiesta di conferma
5. ✅ File salvato

---

**Tutto configurato! Buon lavoro con NIMITZ + vLLM Leitha! 🎉**
