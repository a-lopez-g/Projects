import numpy as np
import pandas as pd
import umap
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from community import community_louvain
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class UMAPLouvainDiscovery:
    """
    Adaptación del método IDS con UMAP + Louvain para descubrimiento de mecanismos en scRNA-seq
    """
    
    def __init__(self, params=None):
        self.params = params or {}
        
        # Parámetros UMAP optimizados para scRNA-seq
        self.umap_params = {
            'n_neighbors': self.params.get('n_neighbors', 15),
            'min_dist': self.params.get('min_dist', 0.1),
            'n_components': self.params.get('n_components', 50),
            'metric': self.params.get('metric', 'cosine'),
            'random_state': 42
        }
        
        # Parámetros para construcción del grafo
        self.graph_params = {
            'similarity_threshold': self.params.get('similarity_threshold', 0.3),
            'k_neighbors': self.params.get('k_neighbors', 10)
        }
        
        # Parámetros Louvain
        self.louvain_params = {
            'resolution': self.params.get('resolution', 1.0),
            'random_state': 42
        }
        
    def preprocess_data(self, X, Y):
        """
        Preprocesamiento específico para scRNA-seq
        """
        print("🔬 Preprocesando datos scRNA-seq...")
        
        # 1. Filtrar genes con muy poca expresión
        gene_expression_sum = np.sum(X, axis=0)
        active_genes = gene_expression_sum > np.percentile(gene_expression_sum, 10)
        X_filtered = X[:, active_genes]
        
        # 2. Normalización log1p (típica en scRNA-seq)
        X_log = np.log1p(X_filtered)
        
        # 3. Separar casos y controles para análisis diferencial
        case_mask = Y == 1
        control_mask = Y == 0
        
        X_cases = X_log[case_mask]
        X_controls = X_log[control_mask]
        
        print(f"   📊 Genes activos: {np.sum(active_genes)}/{len(active_genes)}")
        print(f"   📊 Casos: {np.sum(case_mask)}, Controles: {np.sum(control_mask)}")
        
        return X_log, X_cases, X_controls, active_genes
    
    def compute_differential_features(self, X_cases, X_controls):
        """
        Identificar genes diferencialmente expresados
        """
        print("🔍 Calculando expresión diferencial...")
        
        # Calcular medias por grupo
        mean_cases = np.mean(X_cases, axis=0)
        mean_controls = np.mean(X_controls, axis=0)
        
        # Fold change y significancia estadística simple
        fold_change = mean_cases - mean_controls
        
        # Seleccionar genes con mayor variabilidad diferencial
        abs_fold_change = np.abs(fold_change)
        diff_threshold = np.percentile(abs_fold_change, 75)
        differential_genes = abs_fold_change > diff_threshold
        
        print(f"   📈 Genes diferenciales: {np.sum(differential_genes)}")
        
        return differential_genes, fold_change
    
    def apply_umap_embedding(self, X):
        """
        Aplicar UMAP para reducción dimensional
        """
        print("🗺️  Aplicando embedding UMAP...")
        
        # Normalizar datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Aplicar UMAP
        umap_model = umap.UMAP(**self.umap_params)
        X_embedded = umap_model.fit_transform(X_scaled)
        
        print(f"   📐 Dimensiones: {X.shape} → {X_embedded.shape}")
        
        return X_embedded, umap_model
    
    def build_similarity_graph(self, X_embedded):
        """
        Construir grafo de similitud entre muestras
        """
        print("🕸️  Construyendo grafo de similitud...")
        
        # Calcular similitud coseno
        similarity_matrix = cosine_similarity(X_embedded)
        
        # Crear grafo
        G = nx.Graph()
        n_samples = len(X_embedded)
        
        # Añadir nodos
        for i in range(n_samples):
            G.add_node(i)
        
        # Añadir aristas basadas en similitud
        threshold = self.graph_params['similarity_threshold']
        k_neighbors = self.graph_params['k_neighbors']
        
        for i in range(n_samples):
            # Obtener k vecinos más similares
            similarities = similarity_matrix[i]
            neighbor_indices = np.argsort(similarities)[-k_neighbors-1:-1]  # Excluir self
            
            for j in neighbor_indices:
                if similarities[j] > threshold and i != j:
                    G.add_edge(i, j, weight=similarities[j])
        
        print(f"   🔗 Grafo: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")
        
        return G, similarity_matrix
    
    def detect_communities(self, G):
        """
        Detección de comunidades con Louvain
        """
        print("🏘️  Detectando comunidades con Louvain...")
        
        # Aplicar algoritmo de Louvain
        communities = community_louvain.best_partition(
            G, 
            resolution=self.louvain_params['resolution'],
            random_state=self.louvain_params['random_state']
        )
        
        # Organizar comunidades
        community_dict = defaultdict(list)
        for node, community_id in communities.items():
            community_dict[community_id].append(node)
        
        print(f"   🏘️  Comunidades detectadas: {len(community_dict)}")
        
        return community_dict, communities
    
    def analyze_community_mechanisms(self, community_dict, X, Y, differential_genes, fold_change, active_genes):
        """
        Analizar cada comunidad para identificar mecanismos
        """
        print("🔬 Analizando mecanismos por comunidad...")
        
        discovered_mechanisms = {}
        discovered_assignments = np.zeros((len(X), len(community_dict)))
        
        for comm_id, sample_indices in community_dict.items():
            if len(sample_indices) < 3:  # Filtrar comunidades muy pequeñas
                continue
                
            print(f"   🧬 Analizando comunidad {comm_id} ({len(sample_indices)} muestras)")
            
            # Obtener muestras de la comunidad
            community_samples = X[sample_indices]
            community_labels = Y[sample_indices]
            
            # Calcular perfil de expresión de la comunidad
            community_profile = np.mean(community_samples, axis=0)
            
            # Identificar genes característicos de esta comunidad
            # Comparar con el resto de muestras
            other_indices = [i for i in range(len(X)) if i not in sample_indices]
            other_samples = X[other_indices]
            other_profile = np.mean(other_samples, axis=0)
            
            # Diferencia específica de la comunidad
            community_diff = community_profile - other_profile
            
            # Seleccionar genes más característicos
            abs_diff = np.abs(community_diff)
            top_genes_mask = abs_diff > np.percentile(abs_diff, 85)
            
            # Combinar con genes diferenciales globales
            mechanism_genes_mask = top_genes_mask & differential_genes
            mechanism_gene_indices = np.where(mechanism_genes_mask)[0]
            
            if len(mechanism_gene_indices) > 0:
                # Mapear de vuelta a índices originales
                original_indices = np.where(active_genes)[0][mechanism_gene_indices]
                
                # Efectos (basados en fold change)
                effects = fold_change[mechanism_gene_indices]
                
                # Determinar tipo de mecanismo
                disease_ratio = np.sum(community_labels == 1) / len(community_labels)
                mech_type = 'disease' if disease_ratio > 0.6 else 'common'
                
                discovered_mechanisms[f'mechanism_{comm_id}'] = {
                    'driver_vars': original_indices.tolist(),
                    'effects': effects.tolist(),
                    'type': mech_type,
                    'community_size': len(sample_indices),
                    'disease_ratio': disease_ratio
                }
                
                # Asignaciones
                for sample_idx in sample_indices:
                    discovered_assignments[sample_idx, comm_id] = 1
        
        print(f"   ✅ Mecanismos descubiertos: {len(discovered_mechanisms)}")
        
        return discovered_mechanisms, discovered_assignments
    
    def generate_llm_summaries(self, discovered_mechanisms, X, Y):
        """
        Generar resúmenes tipo LLM para cada mecanismo (simulado)
        """
        print("🤖 Generando resúmenes de mecanismos...")
        
        summaries = {}
        
        for mech_name, mechanism in discovered_mechanisms.items():
            driver_vars = mechanism['driver_vars']
            effects = mechanism['effects']
            mech_type = mechanism['type']
            
            # Simular análisis LLM
            up_regulated = [i for i, eff in enumerate(effects) if eff > 0]
            down_regulated = [i for i, eff in enumerate(effects) if eff < 0]
            
            summary = {
                'mechanism_id': mech_name,
                'type': mech_type,
                'total_genes': len(driver_vars),
                'upregulated_genes': len(up_regulated),
                'downregulated_genes': len(down_regulated),
                'key_genes': driver_vars[:5],  # Top 5 genes
                'biological_interpretation': f"Mecanismo {mech_type} con {len(driver_vars)} genes. "
                                           f"Predomina {'activación' if len(up_regulated) > len(down_regulated) else 'inhibición'}.",
                'confidence_score': mechanism.get('disease_ratio', 0.5)
            }
            
            summaries[mech_name] = summary
        
        return summaries
    
    def discover_mechanisms(self, X, Y, params=None):
        """
        Pipeline principal de descubrimiento
        """
        print("🚀 Iniciando descubrimiento de mecanismos...")
        print("="*60)
        
        # 1. Preprocesamiento
        X_processed, X_cases, X_controls, active_genes = self.preprocess_data(X, Y)
        
        # 2. Análisis diferencial
        differential_genes, fold_change = self.compute_differential_features(X_cases, X_controls)
        
        # 3. UMAP embedding
        X_embedded, umap_model = self.apply_umap_embedding(X_processed)
        
        # 4. Construcción del grafo
        G, similarity_matrix = self.build_similarity_graph(X_embedded)
        
        # 5. Detección de comunidades
        community_dict, communities = self.detect_communities(G)
        
        # 6. Análisis de mecanismos
        discovered_mechanisms, discovered_assignments = self.analyze_community_mechanisms(
            community_dict, X_processed, Y, differential_genes, fold_change, active_genes
        )
        
        # 7. Resúmenes LLM
        mechanism_summaries = self.generate_llm_summaries(discovered_mechanisms, X, Y)
        
        print("="*60)
        print("✅ Descubrimiento completado!")
        
        return {
            'Discovered_Mechanisms': discovered_mechanisms,
            'Discovered_Assignments': discovered_assignments,
            'Mechanism_Summaries': mechanism_summaries,
            'Community_Info': community_dict,
            'UMAP_Embedding': X_embedded,
            'Similarity_Graph': G
        }

# Función wrapper para compatibilidad con el hackathon
def discover_mechanisms(X, Y, params=None):
    """
    Función principal para el hackathon
    """
    discovery_engine = UMAPLouvainDiscovery(params)
    results = discovery_engine.discover_mechanisms(X, Y, params)
    
    return results['Discovered_Mechanisms'], results['Discovered_Assignments']

if __name__ == "__main__":
    # Test con datos sintéticos
    print("🧪 Probando algoritmo de descubrimiento...")
    
    # Cargar datos sintéticos
    try:
        X = np.load('synthetic_data/X_matrix.npy')
        Y = np.load('synthetic_data/Y_labels.npy')
        
        print(f"Datos cargados: X{X.shape}, Y{Y.shape}")
        
        # Ejecutar descubrimiento
        discovered_mechanisms, discovered_assignments = discover_mechanisms(X, Y)
        
        print(f"\n📊 Resultados:")
        print(f"Mecanismos descubiertos: {len(discovered_mechanisms)}")
        print(f"Matriz de asignaciones: {discovered_assignments.shape}")
        
        # Mostrar algunos mecanismos
        for i, (name, mech) in enumerate(list(discovered_mechanisms.items())[:3]):
            print(f"\n🧬 {name}:")
            print(f"   Tipo: {mech['type']}")
            print(f"   Genes: {len(mech['driver_vars'])}")
            print(f"   Primeros genes: {mech['driver_vars'][:5]}")
            
    except FileNotFoundError:
        print("❌ No se encontraron datos sintéticos. Ejecuta primero el generador.")