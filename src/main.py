#!/usr/bin/env python3
"""
NIMITZ
Semantic Image Clustering with Multi-Characteristic Analysis
"""

import os
import numpy as np
import torch
import clip
from PIL import Image
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from itertools import combinations
import json

PROJECT_NAME = "NIMITZ"

class NimitzClusterer:
    """    
    Sistema di clustering semantico per immagini basato su caratteristiche multiple
    configurabili dall'esterno. Come le potenti corazzate Nimitz che dominavano
    i mari, questo sistema domina lo spazio semantico delle immagini.
    """
    
    def __init__(self, model_name="ViT-B/32", device=None):
        """
        Inizializza con configurazione base
        
        Args:
            model_name: Nome del modello CLIP da usare
            device: Device per PyTorch (auto-detect se None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚢 {PROJECT_NAME} deployed on: {self.device}")
        
        # Carica il modello CLIP
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        # Inizializza il sistema di caratteristiche vuoto - da configurare esternamente
        self.characteristics = {}
        
        # Variabili per i dati
        self.image_paths = []
        self.image_features = None
        self.characteristic_features = {}
        self.similarity_matrix = {}
        self.cluster_labels = None
        self.n_clusters = 0
        
        # Inizializza con caratteristiche di default (facilmente sostituibili)
        self._load_default_characteristics()
        
    def _load_default_characteristics(self):
        """Carica un set di caratteristiche di default"""
        self.characteristics = {
            'color_temperature': [
                "warm color palette with reds, oranges, and yellows",
                "cool color palette with blues, greens, and purples", 
                "neutral color palette with grays, beiges, and whites"
            ],
            'color_saturation': [
                "vibrant and saturated colors",
                "muted and desaturated colors",
                "monochromatic color scheme"
            ],
            'lighting_time': [
                "early morning sunrise lighting",
                "bright midday sunlight",
                "afternoon golden hour lighting",
                "evening sunset lighting",
                "nighttime artificial lighting"
            ],
            'lighting_quality': [
                "soft and diffused lighting",
                "harsh and direct lighting",
                "dramatic lighting with strong shadows",
                "even and balanced lighting"
            ]
        }
        
    def add_characteristic(self, name, prompts):
        """
        Aggiunge una nuova caratteristica con i suoi prompt
        
        Args:
            name: Nome della caratteristica
            prompts: Lista di prompt per questa caratteristica
        """
        if not isinstance(prompts, list) or len(prompts) == 0:
            raise ValueError("I prompt devono essere una lista non vuota")
            
        self.characteristics[name] = prompts
        print(f"⚓ {PROJECT_NAME}: Aggiunta caratteristica '{name}' con {len(prompts)} prompt")
        
    def remove_characteristic(self, name):
        """
        Rimuove una caratteristica
        
        Args:
            name: Nome della caratteristica da rimuovere
        """
        if name in self.characteristics:
            del self.characteristics[name]
            print(f"🗑️  {PROJECT_NAME}: Rimossa caratteristica '{name}'")
        else:
            print(f"⚠️  {PROJECT_NAME}: Caratteristica '{name}' non trovata")
            
    def load_characteristics_from_dict(self, characteristics_dict):
        """
        Carica caratteristiche da un dizionario
        
        Args:
            characteristics_dict: Dizionario {nome: [lista_prompt]}
        """
        if not isinstance(characteristics_dict, dict):
            raise ValueError("Le caratteristiche devono essere un dizionario")
            
        self.characteristics = characteristics_dict.copy()
        print(f"📋 {PROJECT_NAME}: Caricate {len(self.characteristics)} caratteristiche dal dizionario")
        
    def load_characteristics_from_json(self, json_path):
        """
        Carica caratteristiche da file JSON
        
        Args:
            json_path: Path al file JSON
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            characteristics = data.get('characteristics', data)
            self.load_characteristics_from_dict(characteristics)
            print(f"📁 {PROJECT_NAME}: Caratteristiche caricate da {json_path}")
            
        except Exception as e:
            print(f"❌ Errore nel caricamento da {json_path}: {e}")
            
    def list_characteristics(self):
        """
        Mostra tutte le caratteristiche disponibili
        """
        print("\n🗂️  {PROJECT_NAME} - ARSENALE DELLE CARATTERISTICHE")
        print("=" * 50)
        
        for name, prompts in self.characteristics.items():
            print(f"\n🎯 {name.upper().replace('_', ' ')}")
            for i, prompt in enumerate(prompts, 1):
                print(f"   {i}. {prompt}")
                
        print(f"\n⚓ Totale: {len(self.characteristics)} categorie di caratteristiche")
        print(f"💪 Potenza di fuoco: {sum(len(p) for p in self.characteristics.values())} prompt")
        
    def get_characteristic_summary(self):
        """
        Ritorna un riepilogo delle caratteristiche configurate
        
        Returns:
            dict: Riepilogo delle caratteristiche
        """
        return {
            'total_categories': len(self.characteristics),
            'total_prompts': sum(len(prompts) for prompts in self.characteristics.values()),
            'categories': list(self.characteristics.keys()),
            'prompts_per_category': {name: len(prompts) for name, prompts in self.characteristics.items()}
        }
        
    def load_images(self, image_directory):
        """
        Carica i percorsi delle immagini da una directory
        
        Args:
            image_directory: Path della directory contenente le immagini
        """
        image_dir = Path(image_directory)
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        self.image_paths = []
        for ext in supported_formats:
            self.image_paths.extend(list(image_dir.glob(f"*{ext}")))
            self.image_paths.extend(list(image_dir.glob(f"*{ext.upper()}")))
            
        print(f"🖼️  {PROJECT_NAME}: Individuate {len(self.image_paths)} immagini da analizzare")
        return self.image_paths
    
    def extract_image_features(self, batch_size=32):
        """
        Estrae le features delle immagini usando CLIP
        
        Args:
            batch_size: Dimensione del batch per il processing
        """
        if not self.image_paths:
            raise ValueError("⚠️  Carica prima le immagini con load_images()")
            
        print("🔍 {PROJECT_NAME}: Scansione delle features delle immagini...")
        
        all_features = []
        valid_paths = []
        
        # Processa le immagini in batch
        for i in tqdm(range(0, len(self.image_paths), batch_size), desc="Scanning images"):
            batch_paths = self.image_paths[i:i+batch_size]
            batch_images = []
            batch_valid_paths = []
            
            # Carica e preprocessa le immagini del batch
            for img_path in batch_paths:
                try:
                    image = Image.open(img_path).convert('RGB')
                    image_tensor = self.preprocess(image).unsqueeze(0)
                    batch_images.append(image_tensor)
                    batch_valid_paths.append(img_path)
                except Exception as e:
                    print(f"⚠️  Errore nel caricare {img_path}: {e}")
                    continue
            
            if batch_images:
                # Stack delle immagini del batch
                batch_tensor = torch.cat(batch_images, dim=0).to(self.device)
                
                # Estrai features con CLIP
                with torch.no_grad():
                    features = self.model.encode_image(batch_tensor)
                    features = features / features.norm(dim=-1, keepdim=True)  # Normalizza
                    
                all_features.append(features.cpu().numpy())
                valid_paths.extend(batch_valid_paths)
        
        if all_features:
            self.image_features = np.vstack(all_features)
            self.image_paths = valid_paths
            print(f"✅ {PROJECT_NAME}: Features estratte per {len(valid_paths)} immagini")
        else:
            raise ValueError("❌ Nessuna feature estratta dalle immagini")
    
    def extract_characteristic_features(self, characteristics_to_use=None):
        """
        Estrae le features per ogni caratteristica separatamente
        
        Args:
            characteristics_to_use: Lista di caratteristiche da usare (None = tutte)
        """
        if not self.characteristics:
            raise ValueError("⚠️  Nessuna caratteristica configurata. Usa add_characteristic() o load_characteristics_from_json()")
            
        if characteristics_to_use is None:
            characteristics_to_use = list(self.characteristics.keys())
        elif isinstance(characteristics_to_use, str):
            characteristics_to_use = [characteristics_to_use]
            
        print(f"🎯 {PROJECT_NAME}: Targeting {len(characteristics_to_use)} caratteristiche...")
        
        self.characteristic_features = {}
        self.characteristic_labels = {}
        
        for char_name in characteristics_to_use:
            if char_name not in self.characteristics:
                print(f"⚠️  Caratteristica '{char_name}' non trovata, saltata")
                continue
                
            prompts = self.characteristics[char_name]
            print(f"   🔍 Processando '{char_name}' ({len(prompts)} prompt)...")
            
            # Tokenizza i prompt per questa caratteristica
            text_tokens = clip.tokenize(prompts).to(self.device)
            
            # Estrai features del testo
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
            self.characteristic_features[char_name] = text_features.cpu().numpy()
            self.characteristic_labels[char_name] = prompts
            
        print(f"⚡ {PROJECT_NAME}: Arsenal ready - {len(self.characteristic_features)} caratteristiche caricate")
            
    def compute_similarity_matrices(self):
        """
        Calcola le matrici di similarità per ogni caratteristica
        """
        if self.image_features is None or not self.characteristic_features:
            raise ValueError("⚠️  Estrai prima le features delle immagini e delle caratteristiche")
            
        print("🧭 {PROJECT_NAME}: Calcolo delle rotte di similarità...")
        
        self.similarity_matrix = {}
        self.similarity_dataframes = {}
        
        for char_name, char_features in self.characteristic_features.items():
            # Calcola la similarità coseno per questa caratteristica
            similarity = np.dot(self.image_features, char_features.T)
            self.similarity_matrix[char_name] = similarity
            
            # Crea DataFrame per analisi più facile
            self.similarity_dataframes[char_name] = pd.DataFrame(
                similarity,
                columns=[f"{char_name}_{i}" for i in range(len(self.characteristic_labels[char_name]))],
                index=[Path(p).name for p in self.image_paths]
            )
            
        print(f"🎯 {PROJECT_NAME}: Calcolate {len(self.similarity_matrix)} matrici di similarità")
    
    def create_combined_feature_space(self, weighting_strategy='equal', custom_weights=None):
        """
        Crea uno spazio delle feature combinato da tutte le caratteristiche
        
        Args:
            weighting_strategy: 'equal', 'variance', 'custom'
            custom_weights: Dict con pesi custom per caratteristica (solo se strategy='custom')
        """
        if not self.similarity_matrix:
            raise ValueError("⚠️  Calcola prima le matrici di similarità")
            
        print("🧩 {PROJECT_NAME}: Costruzione dello spazio semantico unificato...")
        
        # Raccogli tutte le similarità
        all_similarities = []
        feature_names = []
        
        for char_name, similarity in self.similarity_matrix.items():
            all_similarities.append(similarity)
            
            # Crea nomi descrittivi per le feature
            for i, prompt in enumerate(self.characteristic_labels[char_name]):
                # Usa una versione abbreviata del prompt
                short_name = prompt.split()[:3]  # Prime 3 parole
                feature_names.append(f"{char_name}_{' '.join(short_name)}")
        
        # Concatena tutte le similarità
        self.combined_features = np.hstack(all_similarities)
        self.feature_names = feature_names
        
        # Applica pesatura se richiesta
        if weighting_strategy == 'variance':
            # Pesa in base alla varianza di ogni feature
            variances = np.var(self.combined_features, axis=0)
            weights = variances / np.mean(variances)
            self.combined_features = self.combined_features * weights
            print("⚖️  Applicata pesatura basata su varianza")
            
        elif weighting_strategy == 'custom' and custom_weights:
            # Applica pesi custom
            char_names = list(self.similarity_matrix.keys())
            weights = []
            for char_name in char_names:
                char_weight = custom_weights.get(char_name, 1.0)
                n_prompts = len(self.characteristic_labels[char_name])
                weights.extend([char_weight] * n_prompts)
            
            weights = np.array(weights)
            self.combined_features = self.combined_features * weights
            print(f"⚖️  Applicata pesatura personalizzata: {custom_weights}")
            
        print(f"🌊 {PROJECT_NAME}: Spazio unificato creato - {self.combined_features.shape}")
        
    def cluster_images(self, method='kmeans', n_clusters=5, **kwargs):
        """
        Clusterizza le immagini usando lo spazio delle feature combinato
        
        Args:
            method: 'kmeans' o 'dbscan'
            n_clusters: Numero di cluster (solo per kmeans)
            **kwargs: Parametri aggiuntivi per l'algoritmo di clustering
        """
        if self.combined_features is None:
            raise ValueError("⚠️  Crea prima lo spazio delle feature combinato")
            
        print(f"🎲 {PROJECT_NAME}: Attacco con {method.upper()}...")
        
        # Standardizza le features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.combined_features)
        
        # Applica clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, **kwargs)
            cluster_labels = clusterer.fit_predict(features_scaled)
            
        elif method == 'dbscan':
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 2)
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = clusterer.fit_predict(features_scaled)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
        self.cluster_labels = cluster_labels
        self.n_clusters = n_clusters if method == 'kmeans' else n_clusters
        
        print(f"💥 {PROJECT_NAME}: Bombardamento completato - {self.n_clusters} zone conquistate")
        
        # Crea DataFrame con risultati
        self.results_df = pd.DataFrame({
            'image_path': [Path(p).name for p in self.image_paths],
            'cluster': cluster_labels,
            'full_path': [str(p) for p in self.image_paths]
        })
        
        return cluster_labels
    
    def analyze_cluster_characteristics(self, top_k=3):
        """
        Analizza le caratteristiche dominanti per ogni cluster
        
        Args:
            top_k: Numero di caratteristiche top da mostrare per cluster
        """
        if self.combined_features is None or self.cluster_labels is None:
            print("⚠️  Completa prima il clustering")
            return
            
        print("\n📊 {PROJECT_NAME} - RAPPORTO DI INTELLIGENCE")
        print("=" * 60)
        
        cluster_analysis = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_features = self.combined_features[cluster_mask]
            n_images = np.sum(cluster_mask)
            
            print(f"\n🎯 SETTORE {cluster_id} ({n_images} immagini sotto controllo)")
            
            # Calcola feature medie per il cluster
            mean_features = np.mean(cluster_features, axis=0)
            
            # Trova le top-k feature più importanti
            top_indices = np.argsort(mean_features)[-top_k:][::-1]
            
            cluster_chars = []
            for i, feature_idx in enumerate(top_indices):
                feature_name = self.feature_names[feature_idx]
                score = mean_features[feature_idx]
                
                # Separa nome caratteristica e descrizione
                char_name = feature_name.split('_')[0]
                description = ' '.join(feature_name.split('_')[1:])
                
                print(f"  ⚡ {i+1}. {char_name.upper()}: {description}")
                print(f"     🎯 Precisione: {score:.3f}")
                
                cluster_chars.append({
                    'characteristic': char_name,
                    'description': description,
                    'score': score
                })
            
            cluster_analysis[cluster_id] = {
                'n_images': n_images,
                'characteristics': cluster_chars
            }
            
            # Mostra alcune immagini del cluster
            cluster_images = self.results_df[self.results_df['cluster'] == cluster_id]['image_path'].head(3)
            print(f"  📸 Campioni: {', '.join(cluster_images.tolist())}")
            
        self.cluster_analysis = cluster_analysis
        print(f"\n⚓ {PROJECT_NAME}: Intelligence report completato")
        return cluster_analysis
    
    def discover_characteristic_combinations(self, min_score_threshold=0.3):
        """
        Scopre automaticamente combinazioni interessanti di caratteristiche nei cluster
        
        Args:
            min_score_threshold: Soglia minima per considerare una caratteristica significativa
        """
        if not hasattr(self, 'cluster_analysis'):
            print("⚠️  Esegui prima analyze_cluster_characteristics()")
            return
            
        print("\n🔍 {PROJECT_NAME} - SCOPERTA DELLE ROTTE SEMANTICHE")
        print("=" * 50)
        
        discovered_combinations = {}
        
        for cluster_id, analysis in self.cluster_analysis.items():
            chars = analysis['characteristics']
            
            # Filtra caratteristiche significative
            significant_chars = [c for c in chars if c['score'] > min_score_threshold]
            
            if len(significant_chars) >= 2:
                # Crea una descrizione della combinazione
                combination_desc = []
                char_types = []
                
                for char in significant_chars:
                    combination_desc.append(char['description'])
                    char_types.append(char['characteristic'])
                
                combination_key = " + ".join(combination_desc)
                
                discovered_combinations[cluster_id] = {
                    'description': combination_key,
                    'characteristics': char_types,
                    'n_images': analysis['n_images'],
                    'scores': [c['score'] for c in significant_chars]
                }
                
                print(f"\n🌟 Settore {cluster_id}: {combination_key}")
                print(f"   📊 Immagini: {analysis['n_images']}")
                print(f"   🎯 Scores: {[f'{s:.3f}' for s in discovered_combinations[cluster_id]['scores']]}")
        
        self.discovered_combinations = discovered_combinations
        print(f"\n⚓ {PROJECT_NAME}: {len(discovered_combinations)} rotte semantiche scoperte")
        return discovered_combinations
    
    def visualize_clusters(self, save_path=None, show_feature_importance=True):
        """
        Visualizza i cluster e l'importanza delle feature
        """
        if self.combined_features is None or self.cluster_labels is None:
            print("⚠️  Completa prima il clustering")
            return
            
        print("📈 {PROJECT_NAME}: Generazione mappa tattica...")
        
        # Crea subplot
        if show_feature_importance:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot 1: Cluster in spazio PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(self.combined_features)
        
        scatter = ax1.scatter(features_2d[:, 0], features_2d[:, 1], 
                            c=self.cluster_labels, cmap='tab10', alpha=0.7, s=50)
        
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
        ax1.set_title('🚢 {PROJECT_NAME} - Mappa Semantica (PCA 2D)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Settore')
        
        # Plot 2: Importanza delle feature per cluster (se richiesto)
        if show_feature_importance and hasattr(self, 'cluster_analysis'):
            # Crea matrice di importanza
            importance_matrix = np.zeros((self.n_clusters, len(self.characteristics)))
            char_names = list(self.characteristics.keys())
            
            for cluster_id, analysis in self.cluster_analysis.items():
                for char_info in analysis['characteristics']:
                    if char_info['characteristic'] in char_names:
                        char_idx = char_names.index(char_info['characteristic'])
                        importance_matrix[cluster_id, char_idx] = char_info['score']
            
            # Heatmap
            sns.heatmap(importance_matrix, 
                       xticklabels=[c.replace('_', ' ').title() for c in char_names],
                       yticklabels=[f'Settore {i}' for i in range(self.n_clusters)],
                       annot=True, fmt='.3f', cmap='viridis', ax=ax2)
            ax2.set_title('⚡ Potenza di Fuoco per Settore', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Caratteristiche')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Mappa tattica salvata: {save_path}")
        plt.show()
        
    def save_results(self, output_dir="results"):
        """
        Salva tutti i risultati del clustering
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"💾 {PROJECT_NAME}: Archiviazione nel database operativo...")
        
        # 1. Risultati principali
        full_results = self.results_df.copy()
        
        # Aggiungi le caratteristiche top per ogni immagine
        if hasattr(self, 'cluster_analysis'):
            for i, (_, row) in enumerate(full_results.iterrows()):
                cluster_id = row['cluster']
                if cluster_id in self.cluster_analysis and cluster_id != -1:  # -1 è noise in DBSCAN
                    top_char = self.cluster_analysis[cluster_id]['characteristics'][0]
                    full_results.loc[i, 'dominant_characteristic'] = top_char['characteristic']
                    full_results.loc[i, 'dominant_description'] = top_char['description']
                    full_results.loc[i, 'dominant_score'] = top_char['score']
        
        full_results.to_csv(output_path / "clustering_results.csv", index=False)
        
        # 2. Analisi dei cluster
        if hasattr(self, 'cluster_analysis'):
            with open(output_path / "cluster_analysis.json", 'w', encoding='utf-8') as f:
                json.dump(self.cluster_analysis, f, indent=2, default=str, ensure_ascii=False)
        
        # 3. Combinazioni scoperte
        if hasattr(self, 'discovered_combinations'):
            with open(output_path / "discovered_combinations.json", 'w', encoding='utf-8') as f:
                json.dump(self.discovered_combinations, f, indent=2, ensure_ascii=False)
        
        # 4. Configurazione delle caratteristiche usate
        with open(output_path / "characteristics_config.json", 'w', encoding='utf-8') as f:
            json.dump(self.characteristics, f, indent=2, ensure_ascii=False)
        
        # 5. Riepilogo della missione
        summary = {
            'mission_summary': {
                'total_images': len(self.image_paths),
                'total_clusters': self.n_clusters,
                'characteristics_used': list(self.characteristics.keys()),
                'total_prompts': sum(len(prompts) for prompts in self.characteristics.values()),
                'combinations_discovered': len(self.discovered_combinations) if hasattr(self, 'discovered_combinations') else 0
            },
            'characteristic_summary': self.get_characteristic_summary()
        }
        
        with open(output_path / "mission_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📁 {PROJECT_NAME}: Archivio completato in {output_path}")
        print(f"   • clustering_results.csv - Risultati principali")
        print(f"   • cluster_analysis.json - Analisi dettagliata")
        print(f"   • discovered_combinations.json - Combinazioni scoperte")
        print(f"   • characteristics_config.json - Configurazione usata")
        print(f"   • mission_summary.json - Riepilogo missione")
        
        return output_path

    def generate_cluster_cards(self, output_dir=None):
        """
        Genera "carte" in stile trading card per ogni cluster scoperto
        """
        if not hasattr(self, 'cluster_analysis'):
            print("⚠️  Esegui prima analyze_cluster_characteristics()")
            return
            
        if output_dir is None:
            output_dir = "results"
            
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("🃏 {PROJECT_NAME}: Generazione carte collezionabili...")
        
        cards_data = []
        
        for cluster_id, analysis in self.cluster_analysis.items():
            # Calcola statistiche per la "carta"
            cluster_images = self.results_df[self.results_df['cluster'] == cluster_id]
            
            # Top 3 caratteristiche come "stats"
            stats = {}
            for i, char in enumerate(analysis['characteristics'][:3]):
                stat_name = char['characteristic'].replace('_', ' ').title()
                stats[stat_name] = round(char['score'] * 100)  # Converti in percentuale
            
            # Trova immagine rappresentativa (quella più vicina al centroide)
            if len(cluster_images) > 0:
                cluster_mask = self.cluster_labels == cluster_id
                cluster_features = self.combined_features[cluster_mask]
                centroid = np.mean(cluster_features, axis=0)
                
                # Trova l'immagine più vicina al centroide
                distances = np.linalg.norm(cluster_features - centroid, axis=1)
                representative_idx = np.argmin(distances)
                representative_image = cluster_images.iloc[representative_idx]['image_path']
            else:
                representative_image = "N/A"
            
            card_data = {
                'cluster_id': cluster_id,
                'name': f"Settore {cluster_id}",
                'type': 'Semantic Cluster',
                'power_level': sum(stats.values()) // len(stats) if stats else 0,
                'population': analysis['n_images'],
                'representative_image': representative_image,
                'stats': stats,
                'description': f"Cluster caratterizzato da {', '.join([c['characteristic'] for c in analysis['characteristics'][:2]])}"
            }
            
            cards_data.append(card_data)
        
        # Salva i dati delle carte
        with open(output_path / "cluster_cards.json", 'w', encoding='utf-8') as f:
            json.dump(cards_data, f, indent=2, ensure_ascii=False)
        
        print(f"🃏 {PROJECT_NAME}: {len(cards_data)} carte generate e salvate in cluster_cards.json")
        return cards_data

# Utility function per esempi rapidi
def quick_analysis(image_directory, characteristics_config=None, n_clusters=3):
    """
    Funzione di convenience per analisi rapida
    
    Args:
        image_directory: Directory delle immagini
        characteristics_config: Dict o path JSON delle caratteristiche (opzionale)
        n_clusters: Numero di cluster
    """
    print("🚢 {PROJECT_NAME}: Missione rapida iniziata!")
    
    # Inizializza
    nimitz = NimitzClusterer()
    
    # Carica caratteristiche se fornite
    if characteristics_config:
        if isinstance(characteristics_config, str):
            nimitz.load_characteristics_from_json(characteristics_config)
        elif isinstance(characteristics_config, dict):
            nimitz.load_characteristics_from_dict(characteristics_config)
    
    # Pipeline completa
    nimitz.load_images(image_directory)
    nimitz.extract_image_features()
    nimitz.extract_characteristic_features()
    nimitz.compute_similarity_matrices()
    nimitz.create_combined_feature_space()
    nimitz.cluster_images(n_clusters=n_clusters)
    nimitz.analyze_cluster_characteristics()
    nimitz.discover_characteristic_combinations()
    
    print("⚓ {PROJECT_NAME}: Missione rapida completata!")
    return nimitz
