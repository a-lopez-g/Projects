import numpy as np
import json
import time
from pathlib import Path
import argparse
import tracemalloc
import traceback

# Importar módulos del hackathon
from src.usecases.scRNA_data_generator import generate_synthetic_scrna_data
from src.usecases.discovery_umap_louvain import discover_mechanisms as discover_umap_louvain
from src.usecases.discovery_knowledge_graphs import discover_mechanisms as discover_kg
from src.usecases.evaluator import run_evaluation

class BioDiscoveryOrchestrator:
    """
    Orquestador principal que ejecuta todo el pipeline del hackathon
    """
    
    def __init__(self, output_dir="results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Crear subdirectorios
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        
    def generate_test_scenarios(self):
        """
        Definir escenarios de prueba de diferentes complejidades
        """
        scenarios = {
            'simple': {
                'name': 'Escenario Simple',
                'num_samples': 500,
                'num_vars': 1000,
                'mechanism_defs': [
                    {
                        'vars': [10, 25, 47, 89, 156],
                        'effects': [2.5, 1.8, -1.2, 3.0, 1.5],
                        'type': 'disease',
                        'redundancy': 0.3
                    },
                    {
                        'vars': [200, 301, 445],
                        'effects': [1.5, -0.8, 2.2],
                        'type': 'disease',
                        'redundancy': 0.0
                    }
                ],
                'redundancy_level': 0.2
            },
            
            'medium': {
                'name': 'Escenario Medio',
                'num_samples': 1000,
                'num_vars': 2000,
                'mechanism_defs': [
                    {
                        'vars': [10, 25, 47, 89, 156, 234, 345],
                        'effects': [2.5, 1.8, -1.2, 3.0, 1.5, -0.9, 2.1],
                        'type': 'disease',
                        'redundancy': 0.4
                    },
                    {
                        'vars': [500, 601, 745, 823],
                        'effects': [1.5, -0.8, 2.2, -1.1],
                        'type': 'disease',
                        'redundancy': 0.2
                    },
                    {
                        'vars': [1000, 1100, 1200, 1300, 1400],
                        'effects': [0.8, 0.6, -0.4, 0.9, 0.5],
                        'type': 'common',
                        'redundancy': 0.3
                    }
                ],
                'redundancy_level': 0.3
            },
            
            'complex': {
                'name': 'Escenario Complejo',
                'num_samples': 2000,
                'num_vars': 5000,
                'mechanism_defs': [
                    {
                        'vars': [10, 25, 47, 89, 156, 234, 345, 456, 567, 678],
                        'effects': [2.5, 1.8, -1.2, 3.0, 1.5, -0.9, 2.1, -1.3, 1.7, 2.3],
                        'type': 'disease',
                        'redundancy': 0.5
                    },
                    {
                        'vars': [1000, 1101, 1245, 1323, 1456, 1567],
                        'effects': [1.5, -0.8, 2.2, -1.1, 1.8, -1.4],
                        'type': 'disease',
                        'redundancy': 0.3
                    },
                    {
                        'vars': [2000, 2100, 2200, 2300, 2400, 2500, 2600],
                        'effects': [0.8, 0.6, -0.4, 0.9, 0.5, -0.3, 0.7],
                        'type': 'common',
                        'redundancy': 0.4
                    },
                    {
                        'vars': [3000, 3150, 3300, 3450],
                        'effects': [1.2, -0.7, 1.6, -0.9],
                        'type': 'disease',
                        'redundancy': 0.2
                    }
                ],
                'redundancy_level': 0.4
            },
            
            'scalability': {
                'name': 'Escenario Escalabilidad',
                'num_samples': 5000,
                'num_vars': 10000,
                'mechanism_defs': [
                    {
                        'vars': list(range(10, 25)) + list(range(100, 115)),
                        'effects': [np.random.uniform(-2, 3) for _ in range(30)],
                        'type': 'disease',
                        'redundancy': 0.6
                    },
                    {
                        'vars': list(range(1000, 1020)) + list(range(2000, 2010)),
                        'effects': [np.random.uniform(-1.5, 2.5) for _ in range(30)],
                        'type': 'disease',
                        'redundancy': 0.4
                    },
                    {
                        'vars': list(range(5000, 5025)),
                        'effects': [np.random.uniform(-1, 1.5) for _ in range(25)],
                        'type': 'common',
                        'redundancy': 0.5
                    }
                ],
                'redundancy_level': 0.5
            }
        }
        
        return scenarios
    
    def run_scenario(self, scenario_name, scenario_config, algorithms=['umap_louvain', 'knowledge_graphs']):
        """
        Ejecutar un escenario completo
        """
        print(f"\n{'='*80}")
        print(f"🚀 EJECUTANDO: {scenario_config['name']}")
        print(f"{'='*80}")
        
        scenario_dir = self.output_dir / "results" / scenario_name
        scenario_dir.mkdir(exist_ok=True)
        
        results = {
            'scenario': scenario_name,
            'config': scenario_config,
            'algorithms': {}
        }
        
        # 1. GENERAR DATOS
        print("\n📊 FASE 1: Generación de Datos Sintéticos")
        print("-" * 50)
        
        start_time = time.time()
        data = generate_synthetic_scrna_data(
            num_samples=scenario_config['num_samples'],
            num_vars=scenario_config['num_vars'],
            mechanism_defs=scenario_config['mechanism_defs'],
            redundancy_level=scenario_config['redundancy_level']
        )
        generation_time = time.time() - start_time
        
        print(f"✅ Datos generados en {generation_time:.2f} segundos")
        print(f"   📊 Muestras: {data['X'].shape[0]}")
        print(f"   📊 Variables: {data['X'].shape[1]}")
        print(f"   📊 Mecanismos GT: {len(data['M_ground_truth'])}")
        
        # Guardar datos
        data_dir = scenario_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        np.save(data_dir / "X_matrix.npy", data['X'])
        np.save(data_dir / "Y_labels.npy", data['Y'])
        np.save(data_dir / "A_ground_truth.npy", data['A_ground_truth'])
        with open(data_dir / "M_ground_truth.json", 'w') as f:
            json.dump(data['M_ground_truth'], f, indent=4)
        
        # 2. EJECUTAR ALGORITMOS
        for algorithm in algorithms:
            print(f"\n🔬 FASE 2: Algoritmo {algorithm.upper()}")
            print("-" * 50)
            
            # Monitoreo de memoria
            tracemalloc.start()
            start_time = time.time()
            
            try:
                if algorithm == 'umap_louvain':
                    # Parámetros optimizados para UMAP+Louvain
                    params = {
                        'n_neighbors': min(15, scenario_config['num_samples'] // 50),
                        'min_dist': 0.1,
                        'n_components': min(50, scenario_config['num_vars'] // 20),
                        'similarity_threshold': 0.3,
                        'resolution': 1.0
                    }
                    discovered_mechanisms, discovered_assignments = discover_umap_louvain(
                        data['X'], data['Y'], params
                    )
                    
                elif algorithm == 'knowledge_graphs':
                    # Parámetros optimizados para KG
                    params = {
                        'correlation_threshold': 0.5,
                        'n_clusters': None,  # Auto-detect
                        'pathway_weight': 2.0,
                        'ppi_weight': 1.5
                    }
                    discovered_mechanisms, discovered_assignments = discover_kg(
                        data['X'], data['Y'], params
                    )
                
                execution_time = time.time() - start_time
                
                # Memoria utilizada
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                memory_usage = peak / 1024 / 1024  # MB
                
                print(f"✅ {algorithm} completado en {execution_time:.2f} segundos")
                print(f"   💾 Memoria pico: {memory_usage:.2f} MB")
                print(f"   🧬 Mecanismos descubiertos: {len(discovered_mechanisms)}")
                
                # 3. EVALUACIÓN
                print(f"\n📈 FASE 3: Evaluación {algorithm.upper()}")
                print("-" * 50)
                
                evaluation_report = run_evaluation(
                    discovered_mechanisms=discovered_mechanisms,
                    discovered_assignments=discovered_assignments,
                    M_ground_truth=data['M_ground_truth'],
                    A_ground_truth=data['A_ground_truth'],
                    algorithm_name=algorithm,
                    execution_time=execution_time,
                    memory_usage=memory_usage
                )
                
                # Guardar resultados del algoritmo
                algorithm_dir = scenario_dir / algorithm
                algorithm_dir.mkdir(exist_ok=True)
                # Guardar mecanismos descubiertos
                with open(algorithm_dir / "discovered_mechanisms.json", 'w') as f:
                    # Convertir numpy arrays a listas para JSON
                    mechanisms_json = {}
                    for name, mech in discovered_mechanisms.items():
                        mechanisms_json[name] = {
                            k: v.tolist() if isinstance(v, np.ndarray) else v
                            for k, v in mech.items()
                        }
                    json.dump(mechanisms_json, f, indent=4)
                np.save(algorithm_dir / "discovered_assignments.npy", discovered_assignments)
                # Guardar reporte de evaluación
                with open(algorithm_dir / "evaluation_report.json", 'w') as f:
                    report_json = self._convert_numpy_to_json(evaluation_report)
                    json.dump(report_json, f, indent=4)
                
                # Almacenar en resultados
                results['algorithms'][algorithm] = {
                    'execution_time': execution_time,
                    'memory_usage': memory_usage,
                    'mechanisms_found': len(discovered_mechanisms),
                    'global_score': evaluation_report['global_score'],
                    'evaluation_summary': evaluation_report['summary']
                }
                
                print(f"🏆 Score global {algorithm}: {evaluation_report['global_score']:.2f}/100")
                
            except Exception as e:
                print(f"❌ Error ejecutando {algorithm}: {str(e)}")
                print(f"   Detalles: {traceback.format_exc()}")
                
                results['algorithms'][algorithm] = {
                    'error': str(e),
                    'execution_time': None,
                    'memory_usage': None,
                    'global_score': 0
                }
                
                tracemalloc.stop()
        
        # Guardar resumen del escenario
        with open(scenario_dir / "scenario_summary.json", 'w') as f:
            json.dump(self._convert_numpy_to_json(results), f, indent=4)
        
        return results
    
    def _convert_numpy_to_json(self, obj):
        """
        Convertir numpy arrays y tipos a JSON serializable
        """
        if isinstance(obj, dict):
            return {k: self._convert_numpy_to_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_json(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    def run_comparative_analysis(self, scenarios=['simple', 'medium'], algorithms=['umap_louvain', 'knowledge_graphs']):
        """
        Ejecutar análisis comparativo completo
        """
        print("\n" + "="*100)
        print("🏆 HACKATHON BIODISCOVERY - ANÁLISIS COMPARATIVO")
        print("="*100)
        
        all_scenarios = self.generate_test_scenarios()
        comparative_results = {}
        
        for scenario_name in scenarios:
            if scenario_name in all_scenarios:
                scenario_config = all_scenarios[scenario_name]
                results = self.run_scenario(scenario_name, scenario_config, algorithms)
                comparative_results[scenario_name] = results
            else:
                print(f"⚠️  Escenario '{scenario_name}' no encontrado")
        
        # Generar reporte comparativo
        self.generate_comparative_report(comparative_results)
        
        return comparative_results
    
    def generate_comparative_report(self, comparative_results):
        """
        Generar reporte comparativo final
        """
        print("\n" + "="*80)
        print("📊 REPORTE COMPARATIVO FINAL")
        print("="*80)
        
        # Tabla de resultados
        print(f"\n{'Escenario':<15} {'Algoritmo':<20} {'Score':<8} {'Tiempo(s)':<10} {'Memoria(MB)':<12}")
        print("-" * 75)
        
        best_scores = {}
        
        for scenario_name, scenario_results in comparative_results.items():
            for algorithm, alg_results in scenario_results['algorithms'].items():
                if 'error' not in alg_results:
                    score = alg_results['global_score']
                    time_exec = alg_results['execution_time']
                    memory = alg_results['memory_usage']
                    
                    print(f"{scenario_name:<15} {algorithm:<20} {score:<8.2f} {time_exec:<10.2f} {memory:<12.2f}")
                    
                    # Tracking del mejor score por algoritmo
                    if algorithm not in best_scores or score > best_scores[algorithm]['score']:
                        best_scores[algorithm] = {
                            'score': score,
                            'scenario': scenario_name
                        }
                else:
                    print(f"{scenario_name:<15} {algorithm:<20} {'ERROR':<8} {'-':<10} {'-':<12}")
        
        # Resumen de ganadores
        print(f"\n🏆 MEJORES RESULTADOS POR ALGORITMO:")
        print("-" * 50)
        for algorithm, best_result in best_scores.items():
            print(f"{algorithm:<20}: {best_result['score']:.2f} (en {best_result['scenario']})")
        
        # Guardar reporte comparativo
        with open(self.output_dir / "comparative_report.json", 'w') as f:
            json.dump(self._convert_numpy_to_json(comparative_results), f, indent=4)
        
        print(f"\n💾 Reporte completo guardado en: {self.output_dir}")
        print("="*80)

def main():
    """
    Función principal con interfaz de línea de comandos
    """
    parser = argparse.ArgumentParser(description='Hackathon BioDiscovery')
    
    parser.add_argument('--scenarios', nargs='+', 
                       choices=['simple', 'medium', 'complex', 'scalability'],
                       default=['simple', 'medium'],
                       help='Escenarios a ejecutar')
    
    parser.add_argument('--algorithms', nargs='+',
                       choices=['umap_louvain', 'knowledge_graphs'],
                       default=['umap_louvain', 'knowledge_graphs'],
                       help='Algoritmos a comparar')
    
    parser.add_argument('--output-dir', default='hackathon_results',
                       help='Directorio de salida')
    
    args = parser.parse_args()
    
    # Crear orquestador
    orchestrator = BioDiscoveryOrchestrator(output_dir=args.output_dir)
    
    # Ejecutar análisis comparativo
    results = orchestrator.run_comparative_analysis(
        scenarios=args.scenarios,
        algorithms=args.algorithms
    )
    
    print(f"\n✅ Análisis completado. Resultados en: {args.output_dir}")

if __name__ == "__main__":
    main()