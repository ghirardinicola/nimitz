#!/bin/bash
# NIMITZ - Card Informatici Workflow
# Questo script esegue automaticamente il workflow completo

# Don't exit on error immediately, we handle errors manually
set +e

echo "🚢 NIMITZ - Workflow Card Informatici"
echo "======================================"
echo ""

# Load .env file if it exists
if [ -f ".env" ]; then
    echo "📄 Caricamento variabili da .env..."
    set -a  # Automatically export all variables
    source .env
    set +a
    echo "✅ .env caricato"
    echo ""
elif [ -f "../.env" ]; then
    echo "📄 Caricamento variabili da ../.env..."
    set -a
    source ../.env
    set +a
    echo "✅ ../.env caricato"
    echo ""
else
    echo "⚠️  File .env non trovato (le API keys possono essere configurate manualmente)"
    echo ""
fi

# Check if running in interactive mode
if [ -t 0 ]; then
    INTERACTIVE=true
else
    INTERACTIVE=false
fi

# Function to ask yes/no questions
ask_yes_no() {
    if [ "$INTERACTIVE" = false ]; then
        return 0  # Auto-yes in non-interactive mode
    fi
    
    while true; do
        read -p "$1 (y/n): " yn
        case $yn in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Rispondi y o n.";;
        esac
    done
}

# Function to pause and wait for user
pause_step() {
    if [ "$INTERACTIVE" = true ]; then
        echo ""
        read -p "Premi INVIO per continuare..."
        echo ""
    fi
}

echo "======================================"
echo "📋 VERIFICA PREREQUISITI"
echo "======================================"
echo ""

# Check virtual environment (REQUIRED)
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment non attivo"
    # Try common venv names
    venv_activated=false
    for venv_dir in .venv venv env; do
        if [ -d "$venv_dir" ] && [ -f "$venv_dir/bin/activate" ]; then
            echo "Trovato $venv_dir, attivazione in corso..."
            source "$venv_dir/bin/activate"
            echo "✅ $venv_dir attivato"
            venv_activated=true
            break
        fi
    done
    
    if [ "$venv_activated" = false ]; then
        echo "❌ Nessun venv trovato."
        echo "   Crea un venv con: python -m venv .venv"
        echo "   Poi attivalo con: source .venv/bin/activate"
        exit 1
    fi
else
    echo "✅ Virtual environment attivo: $VIRTUAL_ENV"
fi

# Check Python dependencies
echo ""
echo "Verifico dipendenze Python..."

# Check and install if needed
dependencies="requests litellm numpy matplotlib pillow pandas"
missing_deps=""

for dep in $dependencies; do
    # Handle PIL/pillow special case
    if [ "$dep" = "pillow" ]; then
        if ! python -c "from PIL import Image" 2>/dev/null; then
            missing_deps="$missing_deps $dep"
        fi
    else
        if ! python -c "import $dep" 2>/dev/null; then
            missing_deps="$missing_deps $dep"
        fi
    fi
done

if [ -n "$missing_deps" ]; then
    echo "⚠️  Dipendenze mancanti:$missing_deps"
    if ask_yes_no "Vuoi installarle ora?"; then
        echo "Installazione in corso..."
        # Use python -m pip to ensure we install in the correct environment
        if python -m pip --version >/dev/null 2>&1; then
            python -m pip install -q $missing_deps
        else
            # pip module not available, try direct pip from venv
            if [ -f "$VIRTUAL_ENV/bin/pip" ]; then
                $VIRTUAL_ENV/bin/pip install -q $missing_deps
            else
                echo "❌ pip non disponibile. Installa manualmente: pip install$missing_deps"
                exit 1
            fi
        fi
        echo "✅ Dipendenze installate"
    else
        echo "❌ Dipendenze necessarie. Uscita."
        exit 1
    fi
fi

if python -c "import clip" 2>/dev/null; then
    echo "✅ CLIP installato"
else
    echo "⚠️  CLIP non trovato"
    if ask_yes_no "Vuoi installare CLIP ora?"; then
        pip install git+https://github.com/openai/CLIP.git
    fi
fi

if python -c "from src.llm_analyzer import get_llm_config" 2>/dev/null; then
    echo "✅ LLM analyzer disponibile"
else
    echo "⚠️  LLM analyzer non disponibile"
fi

# Check API keys (optional)
echo ""
echo "Verifica API keys (opzionali)..."

if [ -n "$BRAVE_API_KEY" ]; then
    echo "✅ BRAVE_API_KEY configurata"
else
    echo "⚠️  BRAVE_API_KEY non configurata (web research disabilitato)"
    echo "   Ottieni una key gratuita su: https://brave.com/search/api/"
fi

if [ -n "$VLLM_BASE_URL" ] || [ -n "$ANTHROPIC_API_KEY" ] || [ -n "$OPENAI_API_KEY" ] || [ -n "$GEMINI_API_KEY" ]; then
    echo "✅ LLM API configurata"
    [ -n "$VLLM_BASE_URL" ] && echo "   - vLLM: $VLLM_BASE_URL"
    [ -n "$ANTHROPIC_API_KEY" ] && echo "   - Anthropic Claude"
    [ -n "$OPENAI_API_KEY" ] && echo "   - OpenAI"
    [ -n "$GEMINI_API_KEY" ] && echo "   - Google Gemini"
else
    echo "⚠️  Nessuna LLM API configurata (scoring meno accurato)"
    echo "   Configura almeno una di: VLLM_BASE_URL, ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY"
fi

echo ""
echo "======================================"
echo "🎯 STEP 1: Seleziona Informatici"
echo "======================================"
echo ""

if [ -f "informatici.txt" ]; then
    echo "⚠️  File informatici.txt già esistente"
    if ask_yes_no "Vuoi ricreare la lista (sovrascrive)?"; then
        echo ""
        echo "Avvio discovery interattivo..."
        python create_deck_step1.py
    else
        echo "✅ Uso lista esistente"
        echo "   Informatici trovati: $(wc -l < informatici.txt)"
    fi
else
    echo "Avvio discovery interattivo..."
    python create_deck_step1.py
fi

if [ ! -f "informatici.txt" ]; then
    echo "❌ informatici.txt non creato. Uscita."
    exit 1
fi

echo ""
echo "✅ Step 1 completato!"
pause_step

echo "======================================"
echo "🖼️  STEP 2: Scarica Immagini"
echo "======================================"
echo ""

if [ -d "informatici_cards" ] && [ "$(ls -A informatici_cards 2>/dev/null)" ]; then
    echo "⚠️  Directory informatici_cards già contiene file"
    echo "   Immagini trovate: $(ls informatici_cards/*.jpeg 2>/dev/null | wc -l)"
    if ask_yes_no "Vuoi scaricare di nuovo le immagini (sovrascrive)?"; then
        rm -rf informatici_cards
        nimitz retrieve batch informatici.txt --source pexels -o ./informatici_cards
    else
        echo "✅ Uso immagini esistenti"
    fi
else
    echo "Scarico immagini da Pexels..."
    nimitz retrieve batch informatici.txt --source pexels -o ./informatici_cards
    
    # Check if command succeeded (allow partial success)
    if [ ! -d "informatici_cards" ]; then
        echo "❌ Download immagini fallito completamente. Uscita."
        exit 1
    fi
fi

if [ ! -d "informatici_cards" ] || [ -z "$(ls -A informatici_cards 2>/dev/null)" ]; then
    echo "❌ Nessuna immagine scaricata. Uscita."
    exit 1
fi

echo ""
echo "✅ Step 2 completato!"
echo "   Immagini scaricate: $(ls informatici_cards/*.jpeg 2>/dev/null | wc -l)"
pause_step

echo "======================================"
echo "📊 STEP 3a: Vocabolario (Opzionale)"
echo "======================================"
echo ""

if [ -f "vocabolario_informatici_game.json" ]; then
    echo "✅ vocabolario_informatici_game.json trovato"
    echo ""
    echo "Caratteristiche esistenti:"
    cat vocabolario_informatici_game.json | python -c "import sys, json; vocab = json.load(sys.stdin); [print(f'  - {k.replace(\"_\", \" \").title()}') for k in vocab['characteristics'].keys()]" 2>/dev/null || echo "  (impossibile leggere)"
    echo ""
    
    if ask_yes_no "Vuoi modificare il vocabolario?"; then
        nimitz wizard
    else
        echo "✅ Uso vocabolario esistente"
    fi
else
    echo "⚠️  Nessun vocabolario trovato"
    if ask_yes_no "Vuoi creare un vocabolario personalizzato?"; then
        nimitz wizard
    else
        echo "❌ Vocabolario necessario. Uscita."
        exit 1
    fi
fi

echo ""
echo "✅ Step 3a completato!"
pause_step

echo "======================================"
echo "🚀 STEP 3b: Analisi Arricchita"
echo "======================================"
echo ""
echo "Modalità: WEB + CLIP + LLM"
echo ""

if [ -f "informatici_cards_analysis.json" ]; then
    echo "⚠️  File informatici_cards_analysis.json già esistente"
    if ask_yes_no "Vuoi rieseguire l'analisi (sovrascrive)?"; then
        rm informatici_cards_analysis.json
    echo ""
    echo "Avvio analisi arricchita..."
    echo "Questo può richiedere diversi minuti..."
    echo ""
    python create_deck_step3_enriched.py
    
    # Check if analysis succeeded
    if [ $? -ne 0 ]; then
        echo ""
        echo "❌ Analisi fallita. Verifica gli errori sopra."
        exit 1
    fi
else
        echo "✅ Uso analisi esistente"
    fi
else
    echo "Avvio analisi arricchita..."
    echo "Questo può richiedere diversi minuti..."
    echo ""
    python create_deck_step3_enriched.py
fi

if [ ! -f "informatici_cards_analysis.json" ]; then
    echo "❌ Analisi fallita. Uscita."
    exit 1
fi

echo ""
echo "✅ Step 3b completato!"
pause_step

echo ""
echo "======================================"
echo "✅ WORKFLOW COMPLETATO!"
echo "======================================"
echo ""

# Statistics
total_cards=$(cat informatici_cards_analysis.json | python -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "?")

echo "📊 Risultati:"
echo "   Totale carte generate: $total_cards"
echo "   File JSON: informatici_cards_analysis.json"
if [ -d "informatici_cards_visual" ]; then
    echo "   Card grafiche: informatici_cards_visual/"
    echo "      - Pagine: informatici_cards_visual/image_cards_page_*.png"
    echo "      - Individuali: informatici_cards_visual/individual/"
fi
echo ""

if command -v jq &> /dev/null; then
    echo "🏆 Top 5 carte per punteggio medio:"
    echo ""
    cat informatici_cards_analysis.json | jq -r '[.[] | . + {avg: ((.scores.influenza_sul_mercato + .scores.uso_di_tecnologie_avanzate + .scores.riconoscimento_professionale + .scores.lunghezza_della_barba + .scores.openess) / 5)}] | sort_by(.avg) | reverse | .[0:5] | .[] | "  \(.name): \(.avg | floor)/100"' 2>/dev/null || echo "  (impossibile calcolare)"
    echo ""
fi

echo "======================================"
echo "📚 Prossimi Passi"
echo "======================================"
echo ""
echo "Visualizza le card grafiche:"
echo "  open informatici_cards_visual/image_cards_page_1.png"
echo "  open informatici_cards_visual/individual/"
echo ""
echo "Visualizza i dati JSON:"
echo "  cat informatici_cards_analysis.json | jq"
echo ""
echo "Leggi la documentazione:"
echo "  cat WORKFLOW_CARDS_INFORMATICI.md"
echo ""
echo "Verifica statistiche:"
echo "  cat IMPLEMENTATION_SUMMARY.md"
echo ""
echo "======================================"
echo ""
echo "🎮 Buon divertimento con le tue card! ✨"
echo ""
