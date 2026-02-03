#!/bin/bash
#
# Script di configurazione per NIMITZ con vLLM Leitha
#
# Server: agent-codeai.leitha.servizi.gr-u.it
# Modello: Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
#

# Colori per output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  NIMITZ - Configurazione vLLM Leitha  ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Configura le variabili d'ambiente
export VLLM_BASE_URL="https://agent-codeai.leitha.servizi.gr-u.it/v1"
export VLLM_API_KEY="anything"
export VLLM_MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"

echo -e "${GREEN}✓${NC} Variabili d'ambiente configurate:"
echo -e "  VLLM_BASE_URL = ${VLLM_BASE_URL}"
echo -e "  VLLM_API_KEY  = ${VLLM_API_KEY}"
echo -e "  VLLM_MODEL    = ${VLLM_MODEL}"
echo ""

# Test connessione
echo -e "${YELLOW}Testo connessione al server...${NC}"
if curl -s -f -m 5 "${VLLM_BASE_URL%/v1}/health" > /dev/null 2>&1 || \
   curl -s -f -m 5 "${VLLM_BASE_URL}/models" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Server vLLM raggiungibile!"
else
    echo -e "${YELLOW}⚠${NC}  Impossibile verificare la connessione al server"
    echo -e "   Il server potrebbe essere offline o richiedere VPN"
fi
echo ""

# Informazioni sul modello
echo -e "${BLUE}Modello configurato:${NC}"
echo -e "  Nome: Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"
echo -e "  Tipo: Modello di code generation ottimizzato"
echo -e "  Note: Ottimo per task di classificazione e reasoning"
echo ""

# Esempi di utilizzo
echo -e "${BLUE}Esempi di utilizzo:${NC}"
echo ""
echo -e "${GREEN}1. Test rapido del filtering LLM:${NC}"
echo -e "   nimitz retrieve discover \"italian baseball players\" -o test.txt --llm-provider vllm"
echo ""
echo -e "${GREEN}2. Workflow completo (discover + auto-retrieve):${NC}"
echo -e "   nimitz retrieve discover \"famous scientists\" -o scientists.txt --llm-provider vllm --auto"
echo ""
echo -e "${GREEN}3. Usare auto-detection (vLLM ha priorità):${NC}"
echo -e "   nimitz retrieve discover \"query\" -o output.txt"
echo ""

# Aggiungi al profilo (opzionale)
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Vuoi aggiungere la configurazione al tuo ~/.bashrc o ~/.zshrc?${NC}"
echo -e "In questo modo sarà disponibile in ogni terminale."
echo ""
read -p "Aggiungi a bashrc/zshrc? [s/N] " -n 1 -r
echo ""

if [[ $REPLY =~ ^[SsYy]$ ]]; then
    # Determina quale shell profile usare
    if [ -n "$ZSH_VERSION" ]; then
        PROFILE="$HOME/.zshrc"
    elif [ -n "$BASH_VERSION" ]; then
        PROFILE="$HOME/.bashrc"
    else
        PROFILE="$HOME/.profile"
    fi
    
    # Aggiungi la configurazione
    echo "" >> "$PROFILE"
    echo "# NIMITZ vLLM Configuration (Leitha server)" >> "$PROFILE"
    echo "export VLLM_BASE_URL=\"https://agent-codeai.leitha.servizi.gr-u.it/v1\"" >> "$PROFILE"
    echo "export VLLM_API_KEY=\"anything\"" >> "$PROFILE"
    echo "export VLLM_MODEL=\"Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8\"" >> "$PROFILE"
    
    echo -e "${GREEN}✓${NC} Configurazione aggiunta a: $PROFILE"
    echo -e "   Riavvia il terminale o esegui: source $PROFILE"
else
    echo -e "${YELLOW}⚠${NC}  Configurazione NON salvata nel profilo"
    echo -e "   Le variabili sono disponibili solo in questa sessione"
    echo ""
    echo -e "   Per renderle permanenti, aggiungi manualmente a ~/.bashrc o ~/.zshrc:"
    echo -e "   ${BLUE}export VLLM_BASE_URL=\"https://agent-codeai.leitha.servizi.gr-u.it/v1\"${NC}"
    echo -e "   ${BLUE}export VLLM_API_KEY=\"anything\"${NC}"
    echo -e "   ${BLUE}export VLLM_MODEL=\"Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8\"${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Configurazione completata!${NC}"
echo -e "${GREEN}========================================${NC}"
