# NIMITZ

**Trasforma le tue immagini in carte collezionabili con statistiche quantificate.**

Ricordi i **Top Trumps** o i **Quartetti**? Ogni carta descriveva un oggetto e ne quantificava le caratteristiche - velocità, potenza, peso. La USS Nimitz era la carta quasi imbattibile del mazzo "navi".

Ora puoi creare le tue carte, con le tue immagini e le tue caratteristiche.

## Cosa fa

NIMITZ analizza le tue immagini e genera **carte** con statistiche misurabili:

```
┌─────────────────────────────────┐
│  🖼️  foto_tramonto.jpg          │
├─────────────────────────────────┤
│  Temperatura colore  ████████░░ 82  │
│  Saturazione        ███████░░░ 71  │
│  Luce dorata        █████████░ 94  │
│  Composizione       ██████░░░░ 58  │
└─────────────────────────────────┘
```

Tu definisci **quali caratteristiche misurare** - il tuo vocabolario, il tuo data model.

## La visione

NIMITZ vuole essere:

1. **Un generatore di carte** - ogni immagine diventa una carta con stats
2. **Un creatore di vocabolari** - definisci tu le caratteristiche da misurare
3. **Un data model collaborativo** - costruisci insieme al sistema il modo di descrivere le tue immagini

## Quick Start

```bash
# Installa le dipendenze
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git

# Installa nimitz come comando
pip install -e .

# Ora puoi usare nimitz da qualsiasi directory!
nimitz analyze ./foto --preset photography
```

## CLI - Command Line Interface

NIMITZ offre una CLI semplice e intuitiva:

### Analizza una cartella di immagini

```bash
# Analisi con preset fotografia (default)
nimitz analyze ./foto

# Analisi con preset specifico
nimitz analyze ./dipinti --preset art
nimitz analyze ./catalogo --preset products

# Output in directory custom
nimitz analyze ./foto --output ./risultati

# Modalità veloce (senza generazione carte visuali)
nimitz analyze ./foto --no-visual
```

### Descrivi una singola immagine

```bash
# Descrizione base
nimitz describe foto.jpg

# Descrizione dettagliata
nimitz describe foto.jpg --verbose

# Con preset specifico
nimitz describe quadro.jpg --preset art
```

### Visualizza i preset disponibili

```bash
nimitz presets
```

### Crea un vocabolario personalizzato (Wizard)

```bash
# Avvia il wizard interattivo
nimitz wizard

# Wizard con suggerimenti basati sulle tue immagini
nimitz wizard -d ./foto

# Wizard con salvataggio automatico
nimitz wizard -d ./foto -o mio_vocabolario.json

# Wizard + analisi immediata
nimitz wizard -d ./foto --analyze
```

Il wizard ti guida nella creazione di un vocabolario personalizzato:
- **Suggerimenti automatici** basati sulle tue immagini
- **Validazione prompt** ("questo e troppo generico")
- **Ciclo interattivo** per raffinare le caratteristiche
- **Test su immagini** prima di finalizzare

#### Esempio di sessione wizard

```
$ nimitz wizard -d ./foto

============================================================
  NIMITZ - Vocabulary Wizard
  Crea il tuo vocabolario personalizzato
============================================================

 Analizzo le tue immagini per suggerirti un vocabolario...
  Analizzo 8 immagini di esempio...

 Suggerimenti basati sulle tue immagini:
--------------------------------------------------
  ATMOSFERA:
    - atmosfera luminosa e allegra
    - atmosfera scura e misteriosa
    - atmosfera calma e serena

  COMPOSIZIONE:
    - composizione centrata simmetrica
    - composizione con regola dei terzi

Vuoi usare questi suggerimenti come base? [S/n]: s
 Suggerimenti aggiunti!

--------------------------------------------------
Vocabolario corrente: 2 caratteristiche

Cosa vuoi fare? [nuova/modifica/show/test/done]: nuova

 NUOVA CARATTERISTICA
------------------------------
Nome caratteristica (es: 'stile', 'atmosfera'): colore_dominante

Inserisci i prompt per 'colore_dominante'.
(invio vuoto per terminare, minimo 2 prompt)
  1. tonalita calde rosse e arancioni
     Aggiunto!
  2. tonalita fredde blu e verdi
     Aggiunto!
  3. colori
     Prompt troppo corto. Aggiungi piu dettagli...
    Vuoi riprovare? [S/n]: s
  3. palette neutra bianco grigio nero
     Aggiunto!
  4.

 Qualita prompt: 85/100
 Caratteristica 'colore_dominante' aggiunta con 3 prompt!

Cosa vuoi fare? [nuova/modifica/show/test/done]: done

============================================================
  VOCABOLARIO COMPLETATO
============================================================

 Riepilogo:
   Caratteristiche: 3
   Prompt totali: 8

Salva vocabolario su file? [percorso o invio per saltare]: mio_vocab.json
 Salvato in: mio_vocab.json
```

### Valida un vocabolario

```bash
nimitz validate mio_vocabolario.json
```

### Opzioni disponibili

| Comando | Opzione | Descrizione |
|---------|---------|-------------|
| `analyze` | `-p, --preset` | Preset vocabolario (photography, art, products) |
| `analyze` | `-o, --output` | Directory output |
| `analyze` | `--no-visual` | Salta generazione carte PNG (piu veloce) |
| `analyze` | `-q, --quiet` | Output minimale |
| `describe` | `-p, --preset` | Preset vocabolario |
| `describe` | `-v, --verbose` | Output dettagliato |
| `wizard` | `-d, --directory` | Directory immagini per suggerimenti |
| `wizard` | `-o, --output` | File output per salvare vocabolario |
| `wizard` | `--analyze` | Analizza immagini dopo creazione vocabolario |
| `validate` | `vocabulary_file` | File JSON da validare |
| `llm status` | - | Verifica disponibilità provider LLM |
| `llm describe` | `image` | Immagine da analizzare con LLM |
| `llm analyze` | `directory` | Directory da analizzare con LLM |
| `llm vocab` | `directory` | Genera vocabolario con LLM |
| `retrieve status` | - | Verifica configurazione API image retrieval |
| `retrieve single` | `description` | Recupera singola immagine da descrizione |
| `retrieve batch` | `file` | Recupera immagini da file (.txt, .csv, .json) |
| `retrieve discover` | `query` | 🆕 Scopri entità da web search e crea batch file |

## Modalità LLM (senza CLIP)

NIMITZ supporta anche l'analisi tramite LLM multimodali (GPT-4V / Claude Vision / Gemini), senza bisogno di CLIP o PyTorch.

Usa `litellm` come proxy unificato per tutti i provider.

### Setup

```bash
# Installa litellm
pip install litellm

# Imposta la chiave API per il provider che vuoi usare
export ANTHROPIC_API_KEY="your-key"  # per Claude
export GEMINI_API_KEY="your-key"     # per Gemini
export OPENAI_API_KEY="your-key"     # per GPT-4
```

### Provider supportati

| Provider | Modello | Variabile d'ambiente |
|----------|---------|---------------------|
| Anthropic | Claude Sonnet | `ANTHROPIC_API_KEY` |
| Google | Gemini 2.0 Flash | `GEMINI_API_KEY` |
| OpenAI | GPT-4o | `OPENAI_API_KEY` |

### Comandi LLM

```bash
# Verifica quali provider sono disponibili
nimitz llm status

# Descrivi una singola immagine
nimitz llm describe foto.jpg
nimitz llm describe foto.jpg --provider gemini  # usa Gemini
nimitz llm describe foto.jpg --lang it          # risposta in italiano

# Analizza una directory di immagini
nimitz llm analyze ./foto --preset photography
nimitz llm analyze ./foto --provider anthropic  # forza provider specifico

# Genera un vocabolario automaticamente analizzando le tue immagini
nimitz llm vocab ./foto -o mio_vocabolario.json
```

### Vantaggi e svantaggi

| Modalità | Pro | Contro |
|----------|-----|--------|
| CLIP (default) | Veloce, offline, gratuito | Richiede PyTorch, meno flessibile |
| LLM | Più flessibile, descrizioni naturali, genera vocabolari | Costi API, più lento, richiede internet |

## Image Retrieval

Crea carte partendo da **descrizioni testuali**, recuperando automaticamente immagini dal web.

**Esempio d'uso:** vuoi creare un mazzo di carte dei giocatori della tua squadra di baseball? Fornisci solo i nomi, NIMITZ troverà le immagini e genererà le carte!

### Setup

Richiede almeno una API key (entrambe gratuite):

```bash
# Unsplash (raccomandato - ottima qualità, licenza permissiva)
export UNSPLASH_ACCESS_KEY="your-key"
# Registrati su: https://unsplash.com/developers

# O Pexels (alternativa)
export PEXELS_API_KEY="your-key"
# Registrati su: https://www.pexels.com/api/
```

### Comandi

```bash
# Verifica configurazione API
nimitz retrieve status

# Recupera una singola immagine
nimitz retrieve single "Golden Gate Bridge at sunset"
nimitz retrieve single "Marco Bianchi baseball player" --preset art

# Recupera e genera carte da un file di descrizioni
nimitz retrieve batch players.txt                    # File .txt (un nome per riga)
nimitz retrieve batch descriptions.json              # File .json (lista)
nimitz retrieve batch data.csv                       # File .csv

# Opzioni avanzate
nimitz retrieve batch players.txt --source pexels    # Usa Pexels invece di Unsplash
nimitz retrieve batch players.txt --no-clip          # Disabilita selezione CLIP
nimitz retrieve batch players.txt --no-analyze       # Solo scarica, non analizzare
nimitz retrieve batch players.txt --cache ./cache    # Directory cache personalizzata

# 🆕 Web Discovery - Scopri automaticamente i nomi online!
# Richiede BRAVE_API_KEY (2,000 query/mese gratis: https://brave.com/search/api/)
export BRAVE_API_KEY="your-brave-key"

# Trova automaticamente i giocatori e genera il file batch
nimitz retrieve discover "Parma Clima Baseball roster" -o players.txt

# Con template personalizzato per descrizioni più ricche
nimitz retrieve discover "San Francisco Giants 2024" -o giants.txt --template "{name}, baseball player"

# Auto-mode: scopri + recupera immagini + genera carte in un solo comando!
nimitz retrieve discover "Italian Renaissance painters" -o painters.json --auto --preset art
```

### Caratteristiche

- **Multi-source**: Supporto Unsplash e Pexels con licenze permissive
- **CLIP Selection**: Sceglie automaticamente l'immagine più pertinente tra i candidati
- **Smart Caching**: Cache locale per evitare re-download
- **License Tracking**: Metadati completi di licenza e attribuzione per ogni immagine
- **Placeholder Fallback**: Genera immagini placeholder per ricerche fallite
- **Batch Processing**: Processa liste da file .txt, .csv, o .json
- **Full Pipeline**: Integrazione completa con analisi CLIP e generazione carte
- **🆕 Web Discovery**: Scopri automaticamente entità (giocatori, persone, etc.) da ricerche web usando Brave Search

### Esempio completo

```bash
# Crea un file con i nomi dei giocatori
echo "Marco Bianchi, pitcher" > players.txt
echo "Luca Rossi, catcher" >> players.txt
echo "Giovanni Verdi, first base" >> players.txt

# Genera le carte
nimitz retrieve batch players.txt --preset art -o ./baseball_cards

# Output:
# ./baseball_cards/
#   ├── Marco_Bianchi_pitcher.jpg
#   ├── Luca_Rossi_catcher.jpg
#   ├── Giovanni_Verdi_first_base.jpg
#   ├── cards.json
#   ├── cards.csv
#   ├── retrieval_metadata.json
#   └── visual_cards/
```

## Preset vocabolari pronti

NIMITZ include vocabolari pronti per iniziare subito:

```python
from main import quick_analysis

# Per fotografie
results = quick_analysis("./foto", preset="photography")

# Per opere d'arte
results = quick_analysis("./dipinti", preset="art")

# Per immagini prodotto
results = quick_analysis("./catalogo", preset="products")
```

| Preset | Caratteristiche analizzate |
|--------|---------------------------|
| `photography` | composizione, profondità di campo, illuminazione, mood, stile colore, soggetto |
| `art` | stile artistico, medium, palette colori, soggetto, tecnica, tono emotivo |
| `products` | categoria, presentazione, illuminazione, brand feel, schema colori, superfici |

## Definisci il tuo vocabolario

```python
from main import run_nimitz_pipeline

# Le TUE caratteristiche, il TUO modo di descrivere le immagini
mie_caratteristiche = {
    "soggetto": [
        "ritratto di persona",
        "paesaggio naturale",
        "architettura urbana",
        "oggetto in primo piano"
    ],
    "mood": [
        "atmosfera allegra e luminosa",
        "atmosfera malinconica",
        "atmosfera drammatica",
        "atmosfera serena e calma"
    ],
    "stile": [
        "fotografia professionale",
        "scatto spontaneo amatoriale",
        "stile artistico elaborato"
    ]
}

# Genera le carte
results = run_nimitz_pipeline(
    image_directory="./le_mie_foto",
    characteristics=mie_caratteristiche,
    visualize=True
)
```

## Output

NIMITZ genera:
- **Carte visuali con thumbnail** - ogni immagine diventa una carta con anteprima e statistiche
- **Carte individuali** in PNG per ogni immagine
- **Cluster** di immagini simili (il tuo mazzo organizzato)
- **Visualizzazioni** per esplorare la collezione
- **Export CSV/JSON** per usare i dati altrove

## Prossimi passi

Vedi la [Roadmap](ROADMAP.md) completa.

---

*Il nome viene dalla USS Nimitz - la portaerei che era quasi imbattibile nel mazzo "navi" dei Top Trumps.*