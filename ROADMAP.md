# NIMITZ Roadmap

Elenco ordinato delle issue da creare. Ogni issue rappresenta una feature completa.

| # | Issue | Stato |
|---|-------|-------|
| 1 | [MVP Manuale](#1-mvp-manuale) | ✅ done |
| 2 | [Usabilità](#2-usabilità) | ✅ done |
| 3 | [Vocabolario Collaborativo](#3-vocabolario-collaborativo) | ✅ done |
| 4 | [LLM-Only](#4-llm-only) | ✅ done |
| 5 | [Gaming](#5-gaming) | ✅ done |
| 6 | [Image Retrieval](#6-image-retrieval) | ⚪ todo |

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
**Stato:** ⚪ todo

Genera carte partendo da descrizioni testuali, recuperando immagini dal web.

**Esempio d'uso:** creare un mazzo di carte dei giocatori di baseball del Parma Clima partendo solo dai nomi.

### Acceptance Criteria
- [ ] Input da lista testuale (nomi, descrizioni)
- [ ] Ricerca immagini via API (Google Images, Bing, Unsplash, ecc.)
- [ ] Selezione automatica immagine migliore per ogni soggetto
- [ ] Pipeline completa: descrizione → immagine → carta con stats
- [ ] Gestione copyright/licenze immagini
- [ ] Fallback: placeholder per immagini non trovate
- [ ] Batch processing per mazzi completi
- [ ] Cache immagini già scaricate

### Workflow esempio
```
1. Input: "Marco Bianchi, pitcher, Parma Clima Baseball"
2. NIMITZ cerca l'immagine online
3. Analizza l'immagine trovata
4. Genera la carta con statistiche
```

---

Vedi [README](README.md) per iniziare.
