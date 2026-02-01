#!/usr/bin/env python3
"""
NIMITZ - Vocabulary Wizard
Interactive wizard for creating and refining custom vocabularies.
Includes image-based suggestions, prompt validation, and iterative refinement.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

# Try to import CLIP-related modules for image suggestions
try:
    from embed import initialize_clip_model, extract_image_features, extract_text_features
    from core import load_images, validate_characteristics
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


# =============================================================================
# PROMPT VALIDATION
# =============================================================================

# Common patterns that indicate prompts that are too generic
GENERIC_PATTERNS = [
    "good", "bad", "nice", "pretty", "ugly", "beautiful",
    "interesting", "boring", "cool", "amazing", "great",
    "image", "photo", "picture", "thing", "stuff"
]

# Minimum recommended prompt length
MIN_PROMPT_LENGTH = 10

# Maximum recommended prompts per characteristic
MAX_PROMPTS_PER_CHAR = 15
MIN_PROMPTS_PER_CHAR = 2


def validate_prompt(prompt: str) -> Tuple[bool, str]:
    """
    Validate a single prompt for quality.

    Returns:
        Tuple of (is_valid, feedback_message)
    """
    prompt = prompt.strip()

    # Check if empty
    if not prompt:
        return False, "Prompt vuoto. Inserisci una descrizione."

    # Check length
    if len(prompt) < MIN_PROMPT_LENGTH:
        return False, f"Prompt troppo corto ({len(prompt)} caratteri). Aggiungi piu dettagli per aiutare CLIP a identificare questa caratteristica."

    # Check for overly generic words
    prompt_lower = prompt.lower()
    generic_found = [word for word in GENERIC_PATTERNS if word in prompt_lower.split()]
    if generic_found and len(prompt.split()) <= 3:
        return False, f"Prompt troppo generico. Parole come '{', '.join(generic_found)}' sono vaghe. Descrivi specificamente cosa stai cercando."

    # Check for single word
    if len(prompt.split()) == 1:
        return False, "Prompt di una sola parola. CLIP funziona meglio con descrizioni complete come 'paesaggio montano innevato' invece di 'montagna'."

    return True, "OK"


def validate_characteristic_name(name: str, existing_names: List[str]) -> Tuple[bool, str]:
    """
    Validate a characteristic name.

    Returns:
        Tuple of (is_valid, feedback_message)
    """
    name = name.strip()

    if not name:
        return False, "Nome caratteristica vuoto."

    if name.lower() in [n.lower() for n in existing_names]:
        return False, f"Caratteristica '{name}' gia esistente. Scegli un nome diverso."

    if len(name) > 30:
        return False, "Nome troppo lungo. Usa massimo 30 caratteri."

    # Check for invalid characters
    if not all(c.isalnum() or c in '_- ' for c in name):
        return False, "Usa solo lettere, numeri, spazi, trattini e underscore."

    return True, "OK"


def analyze_prompt_quality(prompts: List[str]) -> Dict[str, Any]:
    """
    Analyze the overall quality of a set of prompts.

    Returns:
        Dictionary with quality metrics and suggestions
    """
    if not prompts:
        return {
            'score': 0,
            'issues': ['Nessun prompt definito'],
            'suggestions': ['Aggiungi almeno 2 prompt per questa caratteristica']
        }

    issues = []
    suggestions = []

    # Check number of prompts
    if len(prompts) < MIN_PROMPTS_PER_CHAR:
        issues.append(f"Solo {len(prompts)} prompt. Consigliati almeno {MIN_PROMPTS_PER_CHAR}.")
        suggestions.append("Aggiungi piu varianti per una migliore analisi.")
    elif len(prompts) > MAX_PROMPTS_PER_CHAR:
        issues.append(f"Troppi prompt ({len(prompts)}). Potrebbe rallentare l'analisi.")
        suggestions.append(f"Considera di ridurre a massimo {MAX_PROMPTS_PER_CHAR} prompt.")

    # Check prompt variety
    avg_length = sum(len(p) for p in prompts) / len(prompts)
    if avg_length < 20:
        issues.append("Prompt mediamente molto corti.")
        suggestions.append("Prompt piu descrittivi aiutano CLIP a essere piu preciso.")

    # Check for too similar prompts
    prompt_words = [set(p.lower().split()) for p in prompts]
    for i, words1 in enumerate(prompt_words):
        for j, words2 in enumerate(prompt_words[i+1:], i+1):
            overlap = len(words1 & words2)
            total = len(words1 | words2)
            if total > 0 and overlap / total > 0.7:
                issues.append(f"Prompt {i+1} e {j+1} sono molto simili.")
                suggestions.append("Differenzia i prompt per coprire piu varianti.")
                break

    # Calculate score (0-100)
    score = 100
    score -= len(issues) * 15
    score = max(0, min(100, score))

    return {
        'score': score,
        'issues': issues,
        'suggestions': suggestions,
        'count': len(prompts),
        'avg_length': avg_length
    }


# =============================================================================
# IMAGE-BASED SUGGESTIONS
# =============================================================================

def suggest_from_images(
    image_directory: str,
    max_suggestions: int = 6
) -> Dict[str, List[str]]:
    """
    Analyze sample images and suggest relevant characteristics.

    This uses CLIP to identify what types of content are in the images
    and suggests appropriate vocabulary.
    """
    if not CLIP_AVAILABLE:
        return _get_fallback_suggestions()

    try:
        image_paths = load_images(image_directory)
        if not image_paths:
            print("  Nessuna immagine trovata. Uso suggerimenti generici.")
            return _get_fallback_suggestions()

        # Limit analysis to sample of images
        sample_paths = image_paths[:min(10, len(image_paths))]

        print(f"\n  Analizzo {len(sample_paths)} immagini di esempio...")

        # Initialize CLIP
        model, preprocess, device = initialize_clip_model()

        # Test probes to understand image content
        test_vocabulary = _get_probe_vocabulary()

        # Extract features
        image_features, valid_paths = extract_image_features(
            sample_paths, model, preprocess, device, batch_size=8
        )

        if len(valid_paths) == 0:
            return _get_fallback_suggestions()

        # Test each probe category
        detected_categories = {}

        for category, probes in test_vocabulary.items():
            text_features, _ = extract_text_features({category: probes}, model, device)

            # Compute similarity
            import torch
            with torch.no_grad():
                similarities = (image_features @ text_features[category].T).cpu().numpy()

            # Get average max similarity
            max_sims = similarities.max(axis=1)
            avg_confidence = float(max_sims.mean())

            if avg_confidence > 0.20:  # Threshold for relevance
                detected_categories[category] = avg_confidence

        # Generate suggestions based on detected content
        suggestions = _generate_suggestions_from_detections(detected_categories)

        return suggestions

    except Exception as e:
        print(f"  Errore nell'analisi immagini: {e}")
        return _get_fallback_suggestions()


def _get_probe_vocabulary() -> Dict[str, List[str]]:
    """Probing vocabulary to detect image content types"""
    return {
        'content_type': [
            "photograph of a person",
            "photograph of nature landscape",
            "photograph of architecture building",
            "photograph of food",
            "photograph of product object",
            "artwork painting illustration",
            "abstract art or pattern"
        ],
        'style_type': [
            "professional photography",
            "amateur snapshot",
            "artistic creative image",
            "commercial product shot",
            "documentary photo"
        ],
        'color_mood': [
            "bright colorful vibrant",
            "dark moody atmosphere",
            "black and white monochrome",
            "soft pastel colors",
            "high contrast dramatic"
        ]
    }


def _generate_suggestions_from_detections(detections: Dict[str, float]) -> Dict[str, List[str]]:
    """Generate vocabulary suggestions based on detected content"""
    suggestions = {}

    # Always suggest some basics
    suggestions['qualita_generale'] = [
        "immagine ad alta risoluzione nitida",
        "immagine di media qualita",
        "immagine sfocata o di bassa qualita"
    ]

    # Add mood if relevant
    if detections.get('color_mood', 0) > 0.15:
        suggestions['atmosfera'] = [
            "atmosfera luminosa e allegra",
            "atmosfera scura e misteriosa",
            "atmosfera calma e serena",
            "atmosfera drammatica e intensa"
        ]

    # Add composition
    suggestions['composizione'] = [
        "composizione centrata simmetrica",
        "composizione con regola dei terzi",
        "composizione dinamica diagonale",
        "composizione minimalista"
    ]

    # Content-specific suggestions
    if 'person' in str(detections).lower() or detections.get('content_type', 0) > 0.2:
        suggestions['soggetto'] = [
            "ritratto di persona",
            "gruppo di persone",
            "paesaggio naturale",
            "scena urbana",
            "oggetto in primo piano"
        ]

    return suggestions


def _get_fallback_suggestions() -> Dict[str, List[str]]:
    """Fallback suggestions when image analysis is not available"""
    return {
        'soggetto': [
            "ritratto di persona",
            "paesaggio naturale",
            "architettura urbana",
            "oggetto in primo piano",
            "scena di gruppo"
        ],
        'stile': [
            "fotografia professionale",
            "scatto amatoriale spontaneo",
            "stile artistico elaborato",
            "documentario realistico"
        ],
        'atmosfera': [
            "atmosfera luminosa allegra",
            "atmosfera scura drammatica",
            "atmosfera calma serena",
            "atmosfera energetica dinamica"
        ],
        'colori': [
            "colori vivaci e saturi",
            "toni pastello delicati",
            "bianco e nero",
            "palette calda (rossi, aranci)",
            "palette fredda (blu, verdi)"
        ]
    }


# =============================================================================
# INTERACTIVE WIZARD
# =============================================================================

class VocabularyWizard:
    """Interactive wizard for creating custom vocabularies"""

    def __init__(self, image_directory: Optional[str] = None):
        self.image_directory = image_directory
        self.characteristics: Dict[str, List[str]] = {}
        self.history: List[Dict[str, Any]] = []

    def run(self) -> Dict[str, List[str]]:
        """Run the interactive wizard"""
        self._print_header()

        # Step 1: Image-based suggestions
        if self.image_directory:
            self._offer_suggestions()

        # Step 2: Main editing loop
        self._main_loop()

        # Step 3: Final validation and save
        return self._finalize()

    def _print_header(self):
        """Print wizard header"""
        print("\n" + "=" * 60)
        print("  NIMITZ - Vocabulary Wizard")
        print("  Crea il tuo vocabolario personalizzato")
        print("=" * 60)
        print("\nQuesto wizard ti aiutera a creare un vocabolario")
        print("personalizzato per analizzare le tue immagini.")
        print("\nComandi disponibili durante la creazione:")
        print("  [invio vuoto] - termina inserimento prompt")
        print("  'done'        - termina il wizard")
        print("  'help'        - mostra aiuto")
        print("  'show'        - mostra vocabolario corrente")
        print("  'delete'      - elimina una caratteristica")
        print("  'test'        - testa su immagini di esempio")
        print()

    def _offer_suggestions(self):
        """Offer suggestions based on image analysis"""
        print("\n Analizzo le tue immagini per suggerirti un vocabolario...")

        suggestions = suggest_from_images(self.image_directory)

        if not suggestions:
            print("  Non sono riuscito a generare suggerimenti automatici.")
            return

        print("\n Suggerimenti basati sulle tue immagini:")
        print("-" * 50)

        for char_name, prompts in suggestions.items():
            print(f"\n  {char_name.upper()}:")
            for prompt in prompts:
                print(f"    - {prompt}")

        print("\n" + "-" * 50)
        response = input("\nVuoi usare questi suggerimenti come base? [S/n]: ").strip().lower()

        if response != 'n':
            self.characteristics = suggestions.copy()
            print(" Suggerimenti aggiunti! Puoi modificarli o aggiungerne altri.\n")
        else:
            print(" OK, partiamo da zero.\n")

    def _main_loop(self):
        """Main interaction loop"""
        while True:
            print("\n" + "-" * 50)

            if self.characteristics:
                print(f"Vocabolario corrente: {len(self.characteristics)} caratteristiche")

            action = input("\nCosa vuoi fare? [nuova/modifica/show/test/done]: ").strip().lower()

            if action == 'done' or action == 'fine':
                if not self.characteristics:
                    print(" Attenzione: vocabolario vuoto!")
                    confirm = input("Vuoi uscire comunque? [s/N]: ").strip().lower()
                    if confirm != 's':
                        continue
                break

            elif action == 'help' or action == 'aiuto':
                self._show_help()

            elif action == 'show' or action == 'mostra':
                self._show_current()

            elif action == 'nuova' or action == 'new' or action == 'n':
                self._add_characteristic()

            elif action == 'modifica' or action == 'edit' or action == 'm':
                self._edit_characteristic()

            elif action == 'delete' or action == 'elimina' or action == 'd':
                self._delete_characteristic()

            elif action == 'test' or action == 't':
                self._test_vocabulary()

            elif action == '':
                # Empty input - add new characteristic
                self._add_characteristic()

            else:
                print(f"Comando non riconosciuto: '{action}'")
                print("Usa 'help' per vedere i comandi disponibili.")

    def _show_help(self):
        """Show help text"""
        print("\n AIUTO - Vocabulary Wizard")
        print("=" * 50)
        print("""
Una CARATTERISTICA e' una dimensione di analisi (es: "colore", "stile").
Ogni caratteristica ha dei PROMPT che descrivono i possibili valori.

Esempio:
  Caratteristica: "atmosfera"
  Prompt:
    - "atmosfera allegra e luminosa"
    - "atmosfera malinconica e cupa"
    - "atmosfera calma e rilassata"

CONSIGLI PER BUONI PROMPT:
1. Sii specifico: "luce dorata del tramonto" > "luce"
2. Descrivi visivamente: cosa si VEDE nell'immagine?
3. Evita aggettivi vaghi: "bello", "interessante"
4. Usa 3-8 prompt per caratteristica
5. Fai prompt diversi tra loro (non troppo simili)

COMANDI:
  nuova   - aggiungi nuova caratteristica
  modifica - modifica caratteristica esistente
  show    - mostra vocabolario corrente
  delete  - elimina una caratteristica
  test    - testa su immagini di esempio
  done    - termina e salva
""")

    def _show_current(self):
        """Show current vocabulary"""
        if not self.characteristics:
            print("\n Vocabolario vuoto. Usa 'nuova' per aggiungere caratteristiche.")
            return

        print("\n VOCABOLARIO CORRENTE")
        print("=" * 50)

        for char_name, prompts in self.characteristics.items():
            quality = analyze_prompt_quality(prompts)
            quality_indicator = "" if quality['score'] >= 70 else ""

            print(f"\n {quality_indicator} {char_name.upper()} ({len(prompts)} prompt)")
            for i, prompt in enumerate(prompts, 1):
                print(f"    {i}. {prompt}")

            if quality['issues']:
                print(f"    Suggerimenti: {quality['suggestions'][0] if quality['suggestions'] else ''}")

    def _add_characteristic(self):
        """Add a new characteristic"""
        print("\n NUOVA CARATTERISTICA")
        print("-" * 30)

        # Get name
        while True:
            name = input("Nome caratteristica (es: 'stile', 'atmosfera'): ").strip()

            if name.lower() in ['done', 'fine', 'cancel', 'annulla']:
                print("Aggiunta annullata.")
                return

            valid, message = validate_characteristic_name(name, list(self.characteristics.keys()))
            if valid:
                break
            print(f" {message}")

        # Get prompts
        prompts = []
        print(f"\nInserisci i prompt per '{name}'.")
        print("(invio vuoto per terminare, minimo 2 prompt)")

        while True:
            prompt_num = len(prompts) + 1
            prompt = input(f"  {prompt_num}. ").strip()

            if not prompt:
                if len(prompts) >= MIN_PROMPTS_PER_CHAR:
                    break
                else:
                    print(f"    Inserisci almeno {MIN_PROMPTS_PER_CHAR} prompt.")
                    continue

            valid, message = validate_prompt(prompt)
            if not valid:
                print(f"     {message}")
                retry = input("    Vuoi riprovare? [S/n]: ").strip().lower()
                if retry == 'n':
                    continue
                continue

            prompts.append(prompt)
            print(f"     Aggiunto!")

        # Quality check
        quality = analyze_prompt_quality(prompts)
        print(f"\n Qualita prompt: {quality['score']}/100")

        if quality['issues']:
            print("  Problemi rilevati:")
            for issue in quality['issues']:
                print(f"    - {issue}")

        if quality['score'] < 70:
            improve = input("\nVuoi migliorare i prompt? [s/N]: ").strip().lower()
            if improve == 's':
                prompts = self._improve_prompts(name, prompts)

        self.characteristics[name] = prompts
        print(f"\n Caratteristica '{name}' aggiunta con {len(prompts)} prompt!")

    def _improve_prompts(self, name: str, prompts: List[str]) -> List[str]:
        """Interactive prompt improvement"""
        print(f"\n MIGLIORA PROMPT per '{name}'")
        print("-" * 30)

        while True:
            print("\nPrompt attuali:")
            for i, p in enumerate(prompts, 1):
                print(f"  {i}. {p}")

            print("\nCosa vuoi fare?")
            print("  [numero] - modifica prompt")
            print("  'add'    - aggiungi prompt")
            print("  'del N'  - elimina prompt N")
            print("  'done'   - termina modifiche")

            action = input("> ").strip().lower()

            if action == 'done' or action == 'fine':
                break

            elif action == 'add' or action == 'aggiungi':
                new_prompt = input("Nuovo prompt: ").strip()
                valid, msg = validate_prompt(new_prompt)
                if valid:
                    prompts.append(new_prompt)
                    print(" Aggiunto!")
                else:
                    print(f" {msg}")

            elif action.startswith('del '):
                try:
                    idx = int(action.split()[1]) - 1
                    if 0 <= idx < len(prompts):
                        removed = prompts.pop(idx)
                        print(f" Rimosso: {removed}")
                    else:
                        print(" Indice non valido")
                except (ValueError, IndexError):
                    print(" Uso: del N (es: del 2)")

            elif action.isdigit():
                idx = int(action) - 1
                if 0 <= idx < len(prompts):
                    print(f"Prompt attuale: {prompts[idx]}")
                    new_text = input("Nuovo testo (invio per mantenere): ").strip()
                    if new_text:
                        valid, msg = validate_prompt(new_text)
                        if valid:
                            prompts[idx] = new_text
                            print(" Modificato!")
                        else:
                            print(f" {msg}")
                else:
                    print(" Indice non valido")

        return prompts

    def _edit_characteristic(self):
        """Edit an existing characteristic"""
        if not self.characteristics:
            print(" Nessuna caratteristica da modificare.")
            return

        print("\nCaratteristiche disponibili:")
        names = list(self.characteristics.keys())
        for i, name in enumerate(names, 1):
            print(f"  {i}. {name}")

        choice = input("Quale vuoi modificare? [numero o nome]: ").strip()

        # Find the characteristic
        target = None
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(names):
                target = names[idx]
        elif choice in self.characteristics:
            target = choice

        if target:
            self.characteristics[target] = self._improve_prompts(
                target,
                self.characteristics[target]
            )
        else:
            print(" Caratteristica non trovata.")

    def _delete_characteristic(self):
        """Delete a characteristic"""
        if not self.characteristics:
            print(" Nessuna caratteristica da eliminare.")
            return

        print("\nCaratteristiche disponibili:")
        names = list(self.characteristics.keys())
        for i, name in enumerate(names, 1):
            print(f"  {i}. {name}")

        choice = input("Quale vuoi eliminare? [numero o nome]: ").strip()

        target = None
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(names):
                target = names[idx]
        elif choice in self.characteristics:
            target = choice

        if target:
            confirm = input(f"Confermi eliminazione di '{target}'? [s/N]: ").strip().lower()
            if confirm == 's':
                del self.characteristics[target]
                print(f" '{target}' eliminata.")
        else:
            print(" Caratteristica non trovata.")

    def _test_vocabulary(self):
        """Test vocabulary on sample images"""
        if not self.characteristics:
            print(" Crea prima alcune caratteristiche.")
            return

        if not self.image_directory:
            test_dir = input("Inserisci il percorso delle immagini di test: ").strip()
            if not test_dir:
                return
        else:
            test_dir = self.image_directory

        if not CLIP_AVAILABLE:
            print(" Test non disponibile: CLIP non inizializzato.")
            return

        print(f"\n Testo vocabolario su immagini in: {test_dir}")

        try:
            from main import run_nimitz_pipeline

            results = run_nimitz_pipeline(
                image_directory=test_dir,
                characteristics=self.characteristics,
                visualize=False,
                save_plots=False,
                output_dir="/tmp/nimitz_test"
            )

            print("\n TEST COMPLETATO")
            print(f"  Immagini analizzate: {len(results['image_paths'])}")

            # Show sample results
            if results.get('image_cards_data'):
                print("\n  Esempio risultato:")
                card = results['image_cards_data'][0]
                print(f"    Immagine: {card.get('image_name', 'N/A')}")
                for feat in card.get('dominant_features', [])[:3]:
                    print(f"    - {feat['characteristic']}: {feat['prompt']} ({feat['score']:.2f})")

        except Exception as e:
            print(f" Errore nel test: {e}")

    def _finalize(self) -> Dict[str, List[str]]:
        """Finalize and validate the vocabulary"""
        print("\n" + "=" * 60)
        print("  VOCABOLARIO COMPLETATO")
        print("=" * 60)

        if not self.characteristics:
            print("\nAttenzione: vocabolario vuoto!")
            return {}

        # Final quality report
        total_prompts = sum(len(p) for p in self.characteristics.values())

        print(f"\n Riepilogo:")
        print(f"   Caratteristiche: {len(self.characteristics)}")
        print(f"   Prompt totali: {total_prompts}")

        all_issues = []
        for name, prompts in self.characteristics.items():
            quality = analyze_prompt_quality(prompts)
            if quality['score'] < 70:
                all_issues.append(f"'{name}': {quality['issues'][0] if quality['issues'] else 'qualita bassa'}")

        if all_issues:
            print(f"\n Suggerimenti per migliorare:")
            for issue in all_issues[:3]:
                print(f"   - {issue}")

        # Offer to save
        save_path = input("\nSalva vocabolario su file? [percorso o invio per saltare]: ").strip()
        if save_path:
            try:
                self._save_vocabulary(save_path)
                print(f" Salvato in: {save_path}")
            except Exception as e:
                print(f" Errore salvataggio: {e}")

        return self.characteristics

    def _save_vocabulary(self, path: str):
        """Save vocabulary to JSON file"""
        data = {
            "characteristics": self.characteristics,
            "metadata": {
                "created_by": "NIMITZ Vocabulary Wizard",
                "total_characteristics": len(self.characteristics),
                "total_prompts": sum(len(p) for p in self.characteristics.values())
            }
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# =============================================================================
# CLI ENTRY POINTS
# =============================================================================

def run_wizard(image_directory: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Run the vocabulary wizard.

    Args:
        image_directory: Optional path to images for generating suggestions

    Returns:
        Dictionary of characteristics
    """
    wizard = VocabularyWizard(image_directory)
    return wizard.run()


def quick_validate(characteristics: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Quick validation of a vocabulary.

    Args:
        characteristics: Dictionary of characteristics to validate

    Returns:
        Validation report
    """
    report = {
        'valid': True,
        'total_characteristics': len(characteristics),
        'total_prompts': 0,
        'issues': [],
        'characteristic_quality': {}
    }

    for name, prompts in characteristics.items():
        quality = analyze_prompt_quality(prompts)
        report['characteristic_quality'][name] = quality
        report['total_prompts'] += len(prompts)

        if quality['score'] < 50:
            report['valid'] = False
            report['issues'].append(f"'{name}' ha qualita troppo bassa ({quality['score']}/100)")
        elif quality['score'] < 70:
            report['issues'].append(f"'{name}' potrebbe essere migliorata ({quality['score']}/100)")

    return report


if __name__ == '__main__':
    # Run wizard in standalone mode
    print("NIMITZ Vocabulary Wizard - Standalone Mode")

    if len(sys.argv) > 1:
        image_dir = sys.argv[1]
        print(f"Usando immagini da: {image_dir}")
    else:
        image_dir = None

    vocabulary = run_wizard(image_dir)

    if vocabulary:
        print("\n" + "=" * 50)
        print("VOCABOLARIO FINALE:")
        print(json.dumps(vocabulary, indent=2, ensure_ascii=False))
