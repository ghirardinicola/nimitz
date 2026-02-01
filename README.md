# NIMITZ

**Trasforma le tue immagini in carte collezionabili con statistiche quantificate.**

Come le carte con cui giocavi da bambino - ogni carta descrive un oggetto e ne misura le caratteristiche. NIMITZ era la carta quasi imbattibile del mazzo "navi". Ora puoi creare le tue.

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
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn pillow ftfy regex
pip install git+https://github.com/openai/CLIP.git

# Analizza le tue immagini
cd src
python main.py
```

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
- **Carte individuali** per ogni immagine con tutte le statistiche
- **Cluster** di immagini simili (il tuo mazzo organizzato)
- **Visualizzazioni** per esplorare la collezione
- **Export CSV/JSON** per usare i dati altrove

## Prossimi passi

- [ ] CLI per uso veloce da terminale
- [ ] Creazione vocabolario interattiva
- [ ] Carte visuali con thumbnail dell'immagine
- [ ] Confronto carte (chi vince?)
- [ ] Export in formato carta stampabile

---

*Il nome viene dalla USS Nimitz - la portaerei che era quasi imbattibile nel mazzo "navi".*