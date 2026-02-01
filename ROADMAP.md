# NIMITZ Roadmap

## Fase 1: MVP Manuale
*Funziona, si usa da codice Python*

- [x] Generazione carte con statistiche
- [x] Vocabolario custom via dizionario
- [x] Export CSV/JSON
- [x] Clustering immagini simili
- [ ] Fix: rinominare `exmples.py` → `examples.py`
- [ ] Aggiungere `requirements.txt`
- [ ] Carte visuali con thumbnail (non solo testo)
- [ ] Preset vocabolari pronti (fotografia, arte, prodotti)

## Fase 2: Usabilità
*Più facile da usare*

- [ ] CLI: `nimitz analyze ./foto --preset fotografia`
- [ ] Progress bar durante l'elaborazione
- [ ] Output semplificato (meno file, più chiari)
- [ ] Analisi singola immagine: `nimitz describe foto.jpg`

## Fase 3: Vocabolario Collaborativo
*Il sistema aiuta a costruire il data model*

- [ ] Wizard interattivo per creare vocabolari
- [ ] Suggerimenti basati sulle immagini caricate
- [ ] Validazione prompt ("questo è troppo generico")
- [ ] Ciclo raffina → analizza → migliora

## Fase 4: LLM-Only
*Versione alternativa senza CLIP, usa solo un LLM multimodale*

- [ ] Analisi immagini via GPT-4V / Claude Vision
- [ ] Generazione automatica del vocabolario dall'LLM
- [ ] Scoring caratteristiche via prompt LLM
- [ ] Pro: più flessibile, no setup PyTorch
- [ ] Contro: costi API, più lento

## Fase 5: Gaming
*Le carte diventano un gioco*

- [ ] Confronto carte ("chi vince?")
- [ ] Export carta stampabile (PDF/PNG)
- [ ] Deck builder - costruisci il tuo mazzo
- [ ] Rarità carte basata su unicità features

---

Vedi [README](README.md) per iniziare.
