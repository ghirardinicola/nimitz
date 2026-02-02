# NIMITZ Roadmap

Elenco ordinato delle issue da creare. Ogni issue rappresenta una feature completa.

| # | Issue | Stato |
|---|-------|-------|
| 1 | [MVP Manuale](#1-mvp-manuale) | ✅ done |
| 2 | [Usabilità](#2-usabilità) | ✅ done |
| 3 | [Vocabolario Collaborativo](#3-vocabolario-collaborativo) | ✅ done |
| 4 | [LLM-Only](#4-llm-only) | ✅ done |
| 5 | [Gaming](#5-gaming) | ✅ done |
| 6 | [Image Retrieval](#6-image-retrieval) | ✅ done |

---

## 1. MVP Manuale
**Stato:** ✅ done

Funziona, si usa da codice Python.

### Acceptance Criteria
- [x] Generazione carte con statistiche
- [x] Vocabolario custom via dizionario
- [x] Export CSV/JSON
- [x] Clustering immagini simili
- [x] Fix: rinominare `exmples.py` → `examples.py`
- [x] Aggiungere `requirements.txt`
- [x] Carte visuali con thumbnail (non solo testo)
- [x] Preset vocabolari pronti (fotografia, arte, prodotti)

---

## 2. Usabilità
**Stato:** ✅ done

Più facile da usare.

### Acceptance Criteria
- [x] CLI: `nimitz analyze ./foto --preset fotografia`
- [x] Progress bar durante l'elaborazione
- [x] Output semplificato (meno file, più chiari)
- [x] Analisi singola immagine: `nimitz describe foto.jpg`

---

## 3. Vocabolario Collaborativo
**Stato:** ✅ done

Il sistema aiuta a costruire il data model.

### Acceptance Criteria
- [x] Wizard interattivo per creare vocabolari
- [x] Suggerimenti basati sulle immagini caricate
- [x] Validazione prompt ("questo è troppo generico")
- [x] Ciclo raffina → analizza → migliora

### Nuovi comandi CLI
```bash
# Avvia il wizard interattivo
nimitz wizard

# Wizard con suggerimenti basati su immagini
nimitz wizard -d ./foto

# Wizard con salvataggio automatico
nimitz wizard -d ./foto -o mio_vocabolario.json

# Valida un vocabolario esistente
nimitz validate mio_vocabolario.json
```

---

## 4. LLM-Only
**Stato:** ✅ done

Versione alternativa senza CLIP, usa solo un LLM multimodale.

### Acceptance Criteria
- [x] Analisi immagini via GPT-4V / Claude Vision
- [x] Generazione automatica del vocabolario dall'LLM
- [x] Scoring caratteristiche via prompt LLM

### Note
- Pro: più flessibile, no setup PyTorch
- Contro: costi API, più lento
- Usa `litellm` come proxy unificato per tutti i provider

### Provider supportati
- **Anthropic** (Claude Vision) - `ANTHROPIC_API_KEY`
- **Google** (Gemini) - `GEMINI_API_KEY`
- **OpenAI** (GPT-4V) - `OPENAI_API_KEY`

### Nuovi comandi CLI
```bash
# Verifica disponibilità provider LLM
nimitz llm status

# Descrivi una singola immagine con LLM
nimitz llm describe foto.jpg
nimitz llm describe foto.jpg --provider gemini     # usa Gemini
nimitz llm describe foto.jpg --provider anthropic  # usa Claude
nimitz llm describe foto.jpg --lang it             # risposta in italiano

# Analizza una directory con LLM
nimitz llm analyze ./foto
nimitz llm analyze ./foto --preset art --provider openai

# Genera vocabolario automaticamente con LLM
nimitz llm vocab ./foto -o mio_vocabolario.json
nimitz llm vocab ./foto --samples 10  # usa più immagini di esempio
```

### Requisiti
- Installa litellm: `pip install litellm`
- Imposta almeno una delle variabili d'ambiente:
  - `ANTHROPIC_API_KEY` per Claude
  - `GEMINI_API_KEY` per Gemini
  - `OPENAI_API_KEY` per GPT-4

---

## 5. Gaming
**Stato:** ✅ done

Le carte diventano un gioco.

### Acceptance Criteria
- [x] Confronto carte ("chi vince?")
- [x] Export carta stampabile (PDF/PNG)
- [x] Deck builder - costruisci il tuo mazzo
- [x] Rarità carte basata su unicità features

### Nuovi comandi CLI
```bash
# Confronto carte
nimitz compare cards.json foto1.jpg foto2.jpg        # Confronto power level
nimitz compare cards.json foto1.jpg foto2.jpg -c lighting  # Confronto su caratteristica specifica

# Battaglia completa
nimitz battle cards.json foto1.jpg foto2.jpg        # Battaglia 5 round
nimitz battle cards.json foto1.jpg foto2.jpg -r 3   # Battaglia 3 round

# Rarità
nimitz rarity cards.json                             # Top 10 carte per rarità
nimitz rarity cards.json -n 20                       # Top 20 carte

# Statistiche collezione
nimitz stats cards.json                              # Statistiche complete

# Gestione mazzi
nimitz deck create cards.json --name "My Deck" -o my_deck.json
nimitz deck create cards.json --top 10 -o top10.json  # Top 10 per power
nimitz deck show my_deck.json                        # Info mazzo
nimitz deck show my_deck.json --list                 # Lista tutte le carte
nimitz deck add my_deck.json cards.json foto1.jpg    # Aggiungi carta
nimitz deck remove my_deck.json foto1.jpg            # Rimuovi carta

# Export stampabili
nimitz export-pdf cards.json -o carte.pdf            # Export PDF (3x3 per pagina)
nimitz export-pdf cards.json --size Letter           # Formato Letter
nimitz export-png cards.json -o ./card_images        # Export PNG individuali
```

### Caratteristiche implementate
- **Power Level**: Score complessivo della carta (0-100) basato su max score, mean score e feature ad alta confidenza
- **Rarità**: 5 livelli (Common, Uncommon, Rare, Epic, Legendary) basati sull'unicità della carta rispetto alla collezione
- **Battaglia**: Confronto multi-round su caratteristiche casuali
- **Deck Management**: Crea, salva, carica e modifica mazzi personalizzati
- **Export PDF**: Carte stampabili in formato trading card standard (63.5mm x 88.9mm)

### Requisiti aggiuntivi
- Per PDF export: `pip install reportlab`

---

## 6. Image Retrieval
**Stato:** ✅ done

Genera carte partendo da descrizioni testuali, recuperando immagini dal web.

**Esempio d'uso:** creare un mazzo di carte dei giocatori di baseball del Parma Clima partendo solo dai nomi.

### Acceptance Criteria
- [x] Input da lista testuale (nomi, descrizioni)
- [x] Ricerca immagini via API (Unsplash, Pexels)
- [x] Selezione automatica immagine migliore per ogni soggetto
- [x] Pipeline completa: descrizione → immagine → carta con stats
- [x] Gestione copyright/licenze immagini
- [x] Fallback: placeholder per immagini non trovate
- [x] Batch processing per mazzi completi
- [x] Cache immagini già scaricate

### Workflow esempio
```
1. Input: "Marco Bianchi, pitcher, Parma Clima Baseball"
2. NIMITZ cerca l'immagine online
3. Analizza l'immagine trovata
4. Genera la carta con statistiche
```

### Nuovi comandi CLI
```bash
# Verifica configurazione API
nimitz retrieve status

# Recupera una singola immagine
nimitz retrieve single "Golden Gate Bridge at sunset"
nimitz retrieve single "Marco Bianchi baseball player" --preset art

# Recupera e genera carte da un file di descrizioni
nimitz retrieve batch players.txt                    # File .txt con un nome per riga
nimitz retrieve batch descriptions.json              # File .json con lista
nimitz retrieve batch data.csv                       # File .csv

# Opzioni avanzate
nimitz retrieve batch players.txt --source pexels    # Usa Pexels invece di Unsplash
nimitz retrieve batch players.txt --no-clip          # Disabilita selezione CLIP
nimitz retrieve batch players.txt --no-analyze       # Solo scarica, non analizzare
nimitz retrieve batch players.txt --cache ./cache    # Directory cache custom

# 🆕 Web Discovery - trova automaticamente i nomi online!
nimitz retrieve discover "Parma Clima Baseball roster" -o players.txt
nimitz retrieve discover "San Francisco Giants 2024" -o giants.txt --template "{name}, baseball player"
nimitz retrieve discover "Italian Renaissance painters" -o painters.json --auto  # Auto-retrieve dopo discovery
```

### Requisiti
- **Per Image Retrieval** - API Key richiesta (almeno una):
  - **Unsplash**: `UNSPLASH_ACCESS_KEY` (gratuita, ottima qualità)
    - Registrati su: https://unsplash.com/developers
  - **Pexels**: `PEXELS_API_KEY` (gratuita, ottima qualità)
    - Registrati su: https://www.pexels.com/api/

- **Per Web Discovery** (opzionale):
  - **Brave Search**: `BRAVE_API_KEY` (2,000 query/mese gratis)
    - Registrati su: https://brave.com/search/api/

### Caratteristiche implementate
- **Multi-source**: Supporto Unsplash e Pexels con licenze permissive
- **CLIP Selection**: Sceglie automaticamente l'immagine più pertinente tra i candidati
- **Smart Caching**: Cache locale per evitare re-download
- **License Tracking**: Metadati completi di licenza e attribuzione
- **Placeholder Fallback**: Genera immagini placeholder per ricerche fallite
- **Batch Processing**: Processa liste di descrizioni da file .txt, .csv, o .json
- **Full Pipeline**: Integrazione completa con analisi CLIP e generazione carte
- **🆕 Web Discovery**: Scopre automaticamente entità (giocatori, persone, etc.) da ricerche web con Brave Search

---

Vedi [README](README.md) per iniziare.
