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

# Analizza le tue immagini
cd src
python main.py
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