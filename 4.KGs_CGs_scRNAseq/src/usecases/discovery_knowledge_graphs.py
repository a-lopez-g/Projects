import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import SpectralClustering
from scipy.stats import pearsonr, spearmanr
import json
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class KnowledgeGraphDiscovery:
    """
    Descubrimiento de mecanismos usando Knowledge Graphs (KGs) y Causal Graphs (CGs)
    Versión avanzada que incorpora conocimiento biológico previo
    """
    
    def __init__(self, params=None):
        self.params = params or {}
        
        # Parámetros para construcción del KG
        self.kg_params = {
            'correlation_threshold': self.params.get('correlation_threshold', 0.6),
            'mutual_info_threshold': self.params.get('mutual_info_threshold', 0.3),
            'pathway_weight': self.params.get('pathway_weight', 2.0),
            'ppi_weight': self.params.get('ppi_weight', 1.5)
        }
        
        # Parámetros para clustering espectral
        self.clustering_params = {
            'n_clusters': self.params.get('n_clusters', None),  # Auto-detect
            'gamma': self.params.get('gamma', 1.0),
            'random_state': 42
        }
        
        # Simular conocimiento biológico previo
        self.biological_knowledge = self._initialize_biological_knowledge()
        
    def _initialize_biological_knowledge(self):
        """
        Simular bases de conocimiento biológico (pathways, PPI, etc.)
        En implementación bases como KEGG, STRING, etc.
        """
        return {
            'pathways': {},  # pathway_id -> [gene_list]
            'ppi_network': {},  # gene -> [interacting_genes]
            'gene_ontology': {},  # gene -> [GO_terms]
            'disease_associations': {}  # gene -> [disease_associations]
        }
    
    def preprocess_data(self, X, Y):
        """
        Preprocesamiento avanzado para KG
        """
        print("🔬 Preprocesamiento avanzado para Knowledge Graph...")
        
        # 1. Filtrado más sofisticado
        # Filtrar genes con varianza muy baja
        gene_variance = np.var(X, axis=0)
        high_var_genes = gene_variance > np.percentile(gene_variance, 20)
        
        # Filtrar genes con muy pocos valores no-cero
        non_zero_counts = np.sum(X > 0, axis=0)
        expressed_genes = non_zero_counts > (0.1 * X.shape[0])
        
        # Combinar filtros
        active_genes = high_var_genes & expressed_genes
        X_filtered = X[:, active_genes]
        
        # 2. Normalización robusta
        X_log = np.log1p(X_filtered)
        
        # 3. Separación por grupos
        case_mask = Y == 1
        control_mask = Y == 0
        
        X_cases = X_log[case_mask]
        X_controls = X_log[control_mask]
        
        print(f"   📊 Genes seleccionados: {np.sum(active_genes)}/{len(active_genes)}")
        print(f"   📊 Casos: {np.sum(case_mask)}, Controles: {np.sum(control_mask)}")
        
        return X_log, X_cases, X_controls, active_genes
    
    def compute_gene_correlations(self, X):
        """
        Calcular correlaciones entre genes de manera eficiente
        """
        print("🔗 Calculando correlaciones entre genes...")
        
        n_genes = X.shape[1]
        correlation_matrix = np.corrcoef(X.T)
        
        # Manejar NaN values
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        
        print(f"   📈 Matriz de correlación: {correlation_matrix.shape}")
        
        return correlation_matrix
    
    def build_knowledge_graph(self, X, Y, correlation_matrix, active_genes):
        """
        Construir Knowledge Graph integrando múltiples fuentes
        """
        print("🕸️  Construyendo Knowledge Graph...")
        
        n_genes = X.shape[1]
        original_indices = np.where(active_genes)[0]
        
        # Crear grafo dirigido para capturar relaciones causales
        KG = nx.DiGraph()
        
        # Añadir nodos (genes)
        for i, orig_idx in enumerate(original_indices):
            KG.add_node(i, original_index=orig_idx, gene_id=f"gene_{orig_idx}")
        
        # 1. Aristas basadas en correlación
        corr_threshold = self.kg_params['correlation_threshold']
        for i in range(n_genes):
            for j in range(i+1, n_genes):
                corr_val = abs(correlation_matrix[i, j])
                if corr_val > corr_threshold:
                    # Determinar dirección basada en expresión diferencial
                    case_mean_i = np.mean(X[Y==1, i])
                    case_mean_j = np.mean(X[Y==1, j])
                    control_mean_i = np.mean(X[Y==0, i])
                    control_mean_j = np.mean(X[Y==0, j])
                    
                    diff_i = case_mean_i - control_mean_i
                    diff_j = case_mean_j - control_mean_j
                    
                    # El gen con mayor cambio diferencial "regula" al otro
                    if abs(diff_i) > abs(diff_j):
                        KG.add_edge(i, j, weight=corr_val, type='correlation', strength=corr_val)
                    else:
                        KG.add_edge(j, i, weight=corr_val, type='correlation', strength=corr_val)
        
        # 2. Aristas basadas en conocimiento biológico simulado
        # Simular interacciones proteína-proteína
        ppi_prob = 0.05  # 5% de probabilidad de interacción PPI
        for i in range(n_genes):
            for j in range(i+1, n_genes):
                if np.random.rand() < ppi_prob:
                    ppi_weight = self.kg_params['ppi_weight']
                    if KG.has_edge(i, j):
                        KG[i][j]['weight'] += ppi_weight
                        KG[i][j]['type'] = 'correlation+ppi'
                    elif KG.has_edge(j, i):
                        KG[j][i]['weight'] += ppi_weight
                        KG[j][i]['type'] = 'correlation+ppi'
                    else:
                        KG.add_edge(i, j, weight=ppi_weight, type='ppi', strength=ppi_weight)
        
        # 3. Aristas basadas en pathways simulados
        # Simular genes en el mismo pathway
        pathway_prob = 0.03
        for i in range(n_genes):
            for j in range(i+1, n_genes):
                if np.random.rand() < pathway_prob:
                    pathway_weight = self.kg_params['pathway_weight']
                    if KG.has_edge(i, j):
                        KG[i][j]['weight'] += pathway_weight
                        KG[i][j]['type'] += '+pathway'
                    elif KG.has_edge(j, i):
                        KG[j][i]['weight'] += pathway_weight
                        KG[j][i]['type'] += '+pathway'
                    else:
                        KG.add_edge(i, j, weight=pathway_weight, type='pathway', strength=pathway_weight)
        
        print(f"   🕸️  KG construido: {KG.number_of_nodes()} nodos, {KG.number_of_edges()} aristas")
        
        return KG
    
    def detect_functional_modules(self, KG, X, Y):
        """
        Detectar módulos funcionales usando clustering espectral en el KG
        """
        print("🧩 Detectando módulos funcionales...")
        
        # Convertir a grafo no dirigido para clustering
        UG = KG.to_undirected()
        
        # Crear matriz de adyacencia ponderada
        nodes = list(UG.nodes())
        n_nodes = len(nodes)
        
        if n_nodes < 3:
            print("   ⚠️  Muy pocos nodos para clustering")
            return {}, {}
        
        adjacency_matrix = nx.adjacency_matrix(UG, nodelist=nodes, weight='weight').toarray()
        
        # Auto-detectar número de clusters si no se especifica
        n_clusters = self.clustering_params['n_clusters']
        if n_clusters is None:
            # Usar heurística basada en eigenvalues
            eigenvals = np.linalg.eigvals(adjacency_matrix)
            eigenvals = np.sort(eigenvals)[::-1]
            
            # Buscar el "gap" más grande en eigenvalues
            gaps = np.diff(eigenvals)
            n_clusters = min(np.argmax(gaps) + 2, max(3, n_nodes // 10))
        
        n_clusters = min(n_clusters, n_nodes - 1)
        
        # Aplicar clustering espectral
        try:
            spectral = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                gamma=self.clustering_params['gamma'],
                random_state=self.clustering_params['random_state']
            )
            
            cluster_labels = spectral.fit_predict(adjacency_matrix)
            
            # Organizar en módulos
            modules = defaultdict(list)
            node_to_module = {}
            
            for i, node in enumerate(nodes):
                module_id = cluster_labels[i]
                modules[module_id].append(node)
                node_to_module[node] = module_id
            
            print(f"   🧩 Módulos detectados: {len(modules)}")
            
            return modules, node_to_module
            
        except Exception as e:
            print(f"   ⚠️  Error en clustering espectral: {e}")
            # Fallback: usar componentes conectados
            components = list(nx.connected_components(UG))
            modules = {i: list(comp) for i, comp in enumerate(components)}
            node_to_module = {}
            for i, comp in enumerate(components):
                for node in comp:
                    node_to_module[node] = i
            
            return modules, node_to_module
    
    def analyze_causal_relationships(self, modules, KG, X, Y):
        """
        Analizar relaciones causales dentro de cada módulo
        """
        print("🔍 Analizando relaciones causales...")
        
        causal_mechanisms = {}
        
        for module_id, gene_nodes in modules.items():
            if len(gene_nodes) < 2:
                continue
                
            print(f"   🧬 Analizando módulo {module_id} ({len(gene_nodes)} genes)")
            
            # Extraer subgrafo del módulo
            subgraph = KG.subgraph(gene_nodes).copy()
            
            # Identificar genes "hub" (alta centralidad)
            centrality = nx.degree_centrality(subgraph)
            hub_genes = [node for node, cent in centrality.items() if cent > np.mean(list(centrality.values()))]
            
            # Analizar expresión diferencial del módulo
            module_expression = X[:, gene_nodes]
            case_expression = module_expression[Y == 1]
            control_expression = module_expression[Y == 0]
            
            # Calcular efectos por gen
            effects = []
            original_indices = []
            
            for i, gene_node in enumerate(gene_nodes):
                case_mean = np.mean(case_expression[:, i])
                control_mean = np.mean(control_expression[:, i])
                effect = case_mean - control_mean
                effects.append(effect)
                
                # Obtener índice original del gen
                orig_idx = KG.nodes[gene_node].get('original_index', gene_node)
                original_indices.append(orig_idx)
            
            # Determinar tipo de mecanismo
            case_samples_in_module = self._count_active_samples_in_module(
                module_expression, Y, threshold=np.percentile(np.sum(module_expression, axis=1), 75)
            )
            
            disease_ratio = np.sum((case_samples_in_module > 0) & (Y == 1)) / np.sum(Y == 1)
            mech_type = 'disease' if disease_ratio > 0.5 else 'common'
            
            # Identificar genes conductores vs redundantes
            effect_magnitudes = np.abs(effects)
            driver_threshold = np.percentile(effect_magnitudes, 70)
            
            driver_mask = effect_magnitudes > driver_threshold
            driver_indices = [original_indices[i] for i in range(len(original_indices)) if driver_mask[i]]
            driver_effects = [effects[i] for i in range(len(effects)) if driver_mask[i]]
            
            redundant_indices = [original_indices[i] for i in range(len(original_indices)) if not driver_mask[i]]
            
            if len(driver_indices) > 0:
                causal_mechanisms[f'mechanism_{module_id}'] = {
                    'driver_vars': driver_indices,
                    'redundant_vars': redundant_indices,
                    'effects': driver_effects,
                    'type': mech_type,
                    'hub_genes': [KG.nodes[node].get('original_index', node) for node in hub_genes],
                    'module_size': len(gene_nodes),
                    'disease_ratio': disease_ratio,
                    'causal_edges': list(subgraph.edges(data=True))
                }
        
        print(f"   ✅ Mecanismos causales identificados: {len(causal_mechanisms)}")
        
        return causal_mechanisms
    
    def _count_active_samples_in_module(self, module_expression, Y, threshold):
        """
        Contar muestras activas en un módulo
        """
        module_activity = np.sum(module_expression, axis=1)
        return module_activity > threshold
    
    def assign_samples_to_mechanisms(self, causal_mechanisms, X, Y, active_genes):
        """
        Asignar muestras a mecanismos basado en expresión
        """
        print("📋 Asignando muestras a mecanismos...")
        
        n_samples = X.shape[0]
        n_mechanisms = len(causal_mechanisms)
        assignments = np.zeros((n_samples, n_mechanisms))
        
        mechanism_list = list(causal_mechanisms.keys())
        
        for i, (mech_name, mechanism) in enumerate(causal_mechanisms.items()):
            driver_vars = mechanism['driver_vars']
            
            # Mapear a índices filtrados
            filtered_indices = []
            active_gene_indices = np.where(active_genes)[0]
            
            for driver_var in driver_vars:
                if driver_var in active_gene_indices:
                    filtered_idx = np.where(active_gene_indices == driver_var)[0]
                    if len(filtered_idx) > 0:
                        filtered_indices.append(filtered_idx[0])
            
            if len(filtered_indices) == 0:
                continue
            
            # Calcular actividad del mecanismo por muestra
            mechanism_expression = X[:, filtered_indices]
            mechanism_activity = np.mean(mechanism_expression, axis=1)
            
            # Umbral para asignación
            if mechanism['type'] == 'disease':
                # Para mecanismos de enfermedad, usar percentil alto
                threshold = np.percentile(mechanism_activity, 75)
            else:
                # Para mecanismos comunes, usar percentil medio
                threshold = np.percentile(mechanism_activity, 60)
            
            # Asignar muestras que superan el umbral
            assignments[:, i] = (mechanism_activity > threshold).astype(int)
        
        print(f"   📊 Asignaciones completadas para {n_mechanisms} mecanismos")
        
        return assignments
    
    def generate_biological_summaries(self, causal_mechanisms):
        """
        Generar resúmenes biológicos enriquecidos
        """
        print("🧬 Generando resúmenes biológicos...")
        
        summaries = {}
        
        for mech_name, mechanism in causal_mechanisms.items():
            driver_vars = mechanism['driver_vars']
            effects = mechanism['effects']
            hub_genes = mechanism.get('hub_genes', [])
            causal_edges = mechanism.get('causal_edges', [])
            
            # Análisis de efectos
            up_regulated = [i for i, eff in enumerate(effects) if eff > 0]
            down_regulated = [i for i, eff in enumerate(effects) if eff < 0]
            
            # Análisis de conectividad
            edge_types = Counter([edge[2].get('type', 'unknown') for edge in causal_edges])
            
            summary = {
                'mechanism_id': mech_name,
                'type': mechanism['type'],
                'total_genes': len(driver_vars),
                'hub_genes': hub_genes,
                'upregulated_genes': len(up_regulated),
                'downregulated_genes': len(down_regulated),
                'key_driver_genes': driver_vars[:5],
                'redundant_genes_count': len(mechanism.get('redundant_vars', [])),
                'causal_interactions': len(causal_edges),
                'interaction_types': dict(edge_types),
                'biological_interpretation': self._generate_biological_interpretation(mechanism),
                'confidence_score': mechanism.get('disease_ratio', 0.5),
                'network_properties': {
                    'module_size': mechanism['module_size'],
                    'hub_ratio': len(hub_genes) / len(driver_vars) if driver_vars else 0
                }
            }
            
            summaries[mech_name] = summary
        
        return summaries
    
    def _generate_biological_interpretation(self, mechanism):
        """
        Generar interpretación biológica del mecanismo
        """
        mech_type = mechanism['type']
        n_drivers = len(mechanism['driver_vars'])
        n_redundant = len(mechanism.get('redundant_vars', []))
        effects = mechanism['effects']
        
        up_count = sum(1 for eff in effects if eff > 0)
        down_count = sum(1 for eff in effects if eff < 0)
        
        interpretation = f"Mecanismo {mech_type} con {n_drivers} genes conductores"
        
        if n_redundant > 0:
            interpretation += f" y {n_redundant} genes redundantes"
        
        if up_count > down_count:
            interpretation += ". Predomina la activación transcripcional"
        elif down_count > up_count:
            interpretation += ". Predomina la represión transcripcional"
        else:
            interpretation += ". Balance entre activación e inhibición"
        
        interpretation += f". Ratio enfermedad: {mechanism.get('disease_ratio', 0):.2f}"
        
        return interpretation
    
    def discover_mechanisms(self, X, Y, params=None):
        """
        Pipeline principal de descubrimiento con Knowledge Graphs
        """
        print("🚀 Iniciando descubrimiento con Knowledge Graphs...")
        print("="*70)
        
        # 1. Preprocesamiento
        X_processed, X_cases, X_controls, active_genes = self.preprocess_data(X, Y)
        
        # 2. Calcular correlaciones
        correlation_matrix = self.compute_gene_correlations(X_processed)
        
        # 3. Construir Knowledge Graph
        KG = self.build_knowledge_graph(X_processed, Y, correlation_matrix, active_genes)
        
        # 4. Detectar módulos funcionales
        modules, node_to_module = self.detect_functional_modules(KG, X_processed, Y)
        
        # 5. Analizar relaciones causales
        causal_mechanisms = self.analyze_causal_relationships(modules, KG, X_processed, Y)
        
        # 6. Asignar muestras a mecanismos
        discovered_assignments = self.assign_samples_to_mechanisms(causal_mechanisms, X_processed, Y, active_genes)
        
        # 7. Generar resúmenes biológicos
        biological_summaries = self.generate_biological_summaries(causal_mechanisms)
        
        print("="*70)
        print("✅ Descubrimiento con KG completado!")
        
        return {
            'Discovered_Mechanisms': causal_mechanisms,
            'Discovered_Assignments': discovered_assignments,
            'Biological_Summaries': biological_summaries,
            'Knowledge_Graph': KG,
            'Functional_Modules': modules,
            'Correlation_Matrix': correlation_matrix
        }

# Función wrapper para compatibilidad con el hackathon
def discover_mechanisms(X, Y, params=None):
    """
    Función principal para el hackathon - versión Knowledge Graph
    """
    discovery_engine = KnowledgeGraphDiscovery(params)
    results = discovery_engine.discover_mechanisms(X, Y, params)
    
    return results['Discovered_Mechanisms'], results['Discovered_Assignments']

if __name__ == "__main__":
    # Test con datos sintéticos
    print("🧪 Probando algoritmo KG de descubrimiento...")
    
    # Cargar datos sintéticos
    try:
        X = np.load('synthetic_data/X_matrix.npy')
        Y = np.load('synthetic_data/Y_labels.npy')
        
        print(f"Datos cargados: X{X.shape}, Y{Y.shape}")
        
        # Parámetros específicos para KG
        kg_params = {
            'correlation_threshold': 0.5,
            'n_clusters': None,  # Auto-detect
            'pathway_weight': 2.0,
            'ppi_weight': 1.5
        }
        
        # Ejecutar descubrimiento
        discovered_mechanisms, discovered_assignments = discover_mechanisms(X, Y, kg_params)
        
        print(f"\n📊 Resultados KG:")
        print(f"Mecanismos descubiertos: {len(discovered_mechanisms)}")
        print(f"Matriz de asignaciones: {discovered_assignments.shape}")
        
        # Mostrar algunos mecanismos
        for i, (name, mech) in enumerate(list(discovered_mechanisms.items())[:3]):
            print(f"\n🧬 {name}:")
            print(f"   Tipo: {mech['type']}")
            print(f"   Genes conductores: {len(mech['driver_vars'])}")
            print(f"   Genes redundantes: {len(mech.get('redundant_vars', []))}")
            print(f"   Genes hub: {len(mech.get('hub_genes', []))}")
            print(f"   Primeros conductores: {mech['driver_vars'][:5]}")
            
    except FileNotFoundError:
        print("❌ No se encontraron datos sintéticos. Ejecuta primero el generador.")