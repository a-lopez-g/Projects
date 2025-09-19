import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, adjusted_rand_score
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class MechanismEvaluator:
    """
    Evaluador completo para algoritmos de descubrimiento de mecanismos
    """
    
    def __init__(self):
        self.results = {}
        
    def load_ground_truth(self, M_ground_truth, A_ground_truth):
        """
        Cargar ground truth
        """
        self.M_ground_truth = M_ground_truth
        self.A_ground_truth = A_ground_truth
        
    def evaluate_mechanism_discovery(self, discovered_mechanisms, M_ground_truth):
        """
        Evaluar la calidad del descubrimiento de mecanismos
        """
        print("🔍 Evaluando descubrimiento de mecanismos...")
        
        # Convertir ground truth a formato comparable
        gt_mechanisms = {}
        for mech_name, mech_data in M_ground_truth.items():
            gt_mechanisms[mech_name] = set(mech_data['driver_vars'])
        
        # Convertir discovered mechanisms
        disc_mechanisms = {}
        for mech_name, mech_data in discovered_mechanisms.items():
            disc_mechanisms[mech_name] = set(mech_data['driver_vars'])
        
        # Métricas de descubrimiento
        mechanism_metrics = {}
        
        # 1. Jaccard Index para cada mecanismo
        jaccard_scores = []
        f1_scores = []
        precision_scores = []
        recall_scores = []
        
        # Encontrar la mejor correspondencia entre mecanismos descubiertos y reales
        best_matches = self._find_best_mechanism_matches(gt_mechanisms, disc_mechanisms)
        
        for gt_name, (disc_name, score) in best_matches.items():
            if disc_name is not None:
                gt_genes = gt_mechanisms[gt_name]
                disc_genes = disc_mechanisms[disc_name]
                
                # Jaccard Index
                intersection = len(gt_genes.intersection(disc_genes))
                union = len(gt_genes.union(disc_genes))
                jaccard = intersection / union if union > 0 else 0
                jaccard_scores.append(jaccard)
                
                # Precision, Recall, F1
                if len(disc_genes) > 0:
                    precision = intersection / len(disc_genes)
                    precision_scores.append(precision)
                else:
                    precision_scores.append(0)
                
                if len(gt_genes) > 0:
                    recall = intersection / len(gt_genes)
                    recall_scores.append(recall)
                else:
                    recall_scores.append(0)
                
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                f1_scores.append(f1)
                
                mechanism_metrics[gt_name] = {
                    'matched_with': disc_name,
                    'jaccard': jaccard,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'gt_genes': len(gt_genes),
                    'disc_genes': len(disc_genes),
                    'intersection': intersection
                }
            else:
                # Mecanismo no detectado
                mechanism_metrics[gt_name] = {
                    'matched_with': None,
                    'jaccard': 0,
                    'precision': 0,
                    'recall': 0,
                    'f1': 0,
                    'gt_genes': len(gt_mechanisms[gt_name]),
                    'disc_genes': 0,
                    'intersection': 0
                }
                jaccard_scores.append(0)
                precision_scores.append(0)
                recall_scores.append(0)
                f1_scores.append(0)
        
        # Métricas globales
        global_metrics = {
            'mean_jaccard': np.mean(jaccard_scores),
            'mean_precision': np.mean(precision_scores),
            'mean_recall': np.mean(recall_scores),
            'mean_f1': np.mean(f1_scores),
            'mechanisms_detected': len([m for m in mechanism_metrics.values() if m['matched_with'] is not None]),
            'total_gt_mechanisms': len(gt_mechanisms),
            'total_discovered_mechanisms': len(disc_mechanisms),
            'detection_rate': len([m for m in mechanism_metrics.values() if m['matched_with'] is not None]) / len(gt_mechanisms)
        }
        
        print(f"   📊 Mecanismos detectados: {global_metrics['mechanisms_detected']}/{global_metrics['total_gt_mechanisms']}")
        print(f"   📊 Jaccard promedio: {global_metrics['mean_jaccard']:.3f}")
        print(f"   📊 F1 promedio: {global_metrics['mean_f1']:.3f}")
        
        return {
            'mechanism_metrics': mechanism_metrics,
            'global_metrics': global_metrics,
            'best_matches': best_matches
        }
    
    def _find_best_mechanism_matches(self, gt_mechanisms, disc_mechanisms):
        """
        Encontrar la mejor correspondencia entre mecanismos GT y descubiertos
        """
        best_matches = {}
        used_discovered = set()
        
        # Calcular matriz de similitud
        similarity_matrix = {}
        for gt_name, gt_genes in gt_mechanisms.items():
            similarity_matrix[gt_name] = {}
            for disc_name, disc_genes in disc_mechanisms.items():
                intersection = len(gt_genes.intersection(disc_genes))
                union = len(gt_genes.union(disc_genes))
                jaccard = intersection / union if union > 0 else 0
                similarity_matrix[gt_name][disc_name] = jaccard
        
        # Asignación greedy basada en máxima similitud
        for gt_name in gt_mechanisms.keys():
            best_disc = None
            best_score = 0
            
            for disc_name, score in similarity_matrix[gt_name].items():
                if disc_name not in used_discovered and score > best_score:
                    best_disc = disc_name
                    best_score = score
            
            if best_score > 0.1:  # Umbral mínimo de similitud
                best_matches[gt_name] = (best_disc, best_score)
                used_discovered.add(best_disc)
            else:
                best_matches[gt_name] = (None, 0)
        
        return best_matches
    
    def evaluate_sample_assignments(self, discovered_assignments, A_ground_truth):
        """
        Evaluar la calidad de las asignaciones de muestras
        """
        print("📋 Evaluando asignaciones de muestras...")
        
        n_samples, n_gt_mechanisms = A_ground_truth.shape
        n_disc_mechanisms = discovered_assignments.shape[1]
        
        assignment_metrics = {}
        
        # 1. Accuracy por mecanismo (si hay correspondencia 1:1)
        min_mechanisms = min(n_gt_mechanisms, n_disc_mechanisms)
        
        mechanism_accuracies = []
        mechanism_f1s = []
        
        for i in range(min_mechanisms):
            gt_assignments = A_ground_truth[:, i]
            disc_assignments = discovered_assignments[:, i]
            
            # Accuracy
            accuracy = accuracy_score(gt_assignments, disc_assignments)
            mechanism_accuracies.append(accuracy)
            
            # F1 Score
            f1 = f1_score(gt_assignments, disc_assignments, average='binary', zero_division=0)
            mechanism_f1s.append(f1)
            
            assignment_metrics[f'mechanism_{i}'] = {
                'accuracy': accuracy,
                'f1': f1,
                'gt_positive': np.sum(gt_assignments),
                'disc_positive': np.sum(disc_assignments)
            }
        
        # 2. Adjusted Rand Index (para clustering)
        # Convertir matrices binarias a etiquetas de cluster
        gt_labels = self._binary_matrix_to_cluster_labels(A_ground_truth)
        disc_labels = self._binary_matrix_to_cluster_labels(discovered_assignments)
        
        ari_score = adjusted_rand_score(gt_labels, disc_labels)
        
        # 3. Métricas globales
        global_assignment_metrics = {
            'mean_accuracy': np.mean(mechanism_accuracies),
            'mean_f1': np.mean(mechanism_f1s),
            'adjusted_rand_index': ari_score,
            'mechanisms_compared': min_mechanisms,
            'gt_mechanisms': n_gt_mechanisms,
            'discovered_mechanisms': n_disc_mechanisms
        }
        
        print(f"   📊 Accuracy promedio: {global_assignment_metrics['mean_accuracy']:.3f}")
        print(f"   📊 F1 promedio: {global_assignment_metrics['mean_f1']:.3f}")
        print(f"   📊 Adjusted Rand Index: {global_assignment_metrics['adjusted_rand_index']:.3f}")
        
        return {
            'assignment_metrics': assignment_metrics,
            'global_assignment_metrics': global_assignment_metrics
        }
    
    def _binary_matrix_to_cluster_labels(self, binary_matrix):
        """
        Convertir matriz binaria de asignaciones a etiquetas de cluster
        """
        n_samples = binary_matrix.shape[0]
        cluster_labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            # Encontrar mecanismos activos para esta muestra
            active_mechanisms = np.where(binary_matrix[i, :] == 1)[0]
            
            if len(active_mechanisms) == 0:
                cluster_labels[i] = 0  # Sin mecanismo
            elif len(active_mechanisms) == 1:
                cluster_labels[i] = active_mechanisms[0] + 1
            else:
                # Múltiples mecanismos: crear etiqueta combinada
                cluster_labels[i] = hash(tuple(sorted(active_mechanisms))) % 1000 + 1000
        
        return cluster_labels
    
    def compute_computational_metrics(self, algorithm_name, execution_time, memory_usage=None):
        """
        Calcular métricas computacionales
        """
        print("⚡ Evaluando eficiencia computacional...")
        
        computational_metrics = {
            'algorithm': algorithm_name,
            'execution_time_seconds': execution_time,
            'execution_time_minutes': execution_time / 60,
            'memory_usage_mb': memory_usage,
            'efficiency_score': self._calculate_efficiency_score(execution_time, memory_usage)
        }
        
        print(f"   ⏱️  Tiempo de ejecución: {execution_time:.2f} segundos")
        if memory_usage:
            print(f"   💾 Uso de memoria: {memory_usage:.2f} MB")
        
        return computational_metrics
    
    def _calculate_efficiency_score(self, execution_time, memory_usage):
        """
        Calcular score de eficiencia (0-100)
        """
        # Score basado en tiempo (penalizar tiempos > 300 segundos)
        time_score = max(0, 100 - (execution_time / 300) * 50)
        
        # Score basado en memoria (si disponible)
        if memory_usage:
            memory_score = max(0, 100 - (memory_usage / 1000) * 30)  # Penalizar > 1GB
            return (time_score + memory_score) / 2
        else:
            return time_score
    
    def generate_evaluation_report(self, mechanism_results, assignment_results, computational_results=None):
        """
        Generar reporte completo de evaluación
        """
        print("📊 Generando reporte de evaluación...")
        
        report = {
            'summary': {
                'mechanism_discovery': {
                    'detection_rate': mechanism_results['global_metrics']['detection_rate'],
                    'mean_jaccard': mechanism_results['global_metrics']['mean_jaccard'],
                    'mean_f1': mechanism_results['global_metrics']['mean_f1']
                },
                'sample_assignment': {
                    'mean_accuracy': assignment_results['global_assignment_metrics']['mean_accuracy'],
                    'mean_f1': assignment_results['global_assignment_metrics']['mean_f1'],
                    'adjusted_rand_index': assignment_results['global_assignment_metrics']['adjusted_rand_index']
                }
            },
            'detailed_results': {
                'mechanism_discovery': mechanism_results,
                'sample_assignment': assignment_results
            }
        }
        
        if computational_results:
            report['summary']['computational_efficiency'] = {
                'execution_time': computational_results['execution_time_seconds'],
                'efficiency_score': computational_results['efficiency_score']
            }
            report['detailed_results']['computational'] = computational_results
        
        # Calcular score global
        global_score = self._calculate_global_score(report['summary'])
        report['global_score'] = global_score
        
        print(f"   🏆 Score global: {global_score:.2f}/100")
        
        return report
    
    def _calculate_global_score(self, summary):
        """
        Calcular score global del algoritmo
        """
        # Pesos según criterios del hackathon
        weights = {
            'performance': 0.4,  # 40%
            'robustness': 0.2,   # 20%
            'innovation': 0.2,   # 20% (no evaluable automáticamente)
            'simplicity': 0.2    # 20% (no evaluable automáticamente)
        }
        
        # Score de performance (40%)
        mech_f1 = summary['mechanism_discovery']['mean_f1']
        assign_f1 = summary['sample_assignment']['mean_f1']
        performance_score = (mech_f1 + assign_f1) / 2 * 100
        
        # Score de robustez (20%) - basado en detección y consistencia
        detection_rate = summary['mechanism_discovery']['detection_rate']
        ari = summary['sample_assignment']['adjusted_rand_index']
        robustness_score = (detection_rate + max(0, ari)) / 2 * 100
        
        # Scores de innovación y simplicidad se asumen como 70 (promedio)
        innovation_score = 70
        simplicity_score = 70
        
        # Score computacional (bonus/penalty)
        computational_bonus = 0
        if 'computational_efficiency' in summary:
            eff_score = summary['computational_efficiency']['efficiency_score']
            computational_bonus = (eff_score - 50) * 0.1  # ±5 puntos max
        
        global_score = (
            performance_score * weights['performance'] +
            robustness_score * weights['robustness'] +
            innovation_score * weights['innovation'] +
            simplicity_score * weights['simplicity'] +
            computational_bonus
        )
        
        return min(100, max(0, global_score))
    
    def visualize_results(self, mechanism_results, assignment_results, save_plots=True):
        """
        Generar visualizaciones de los resultados
        """
        print("📈 Generando visualizaciones...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Evaluación del Algoritmo de Descubrimiento de Mecanismos', fontsize=16)
        
        # 1. Métricas de descubrimiento de mecanismos
        mech_metrics = mechanism_results['mechanism_metrics']
        mech_names = list(mech_metrics.keys())
        jaccard_scores = [mech_metrics[name]['jaccard'] for name in mech_names]
        f1_scores = [mech_metrics[name]['f1'] for name in mech_names]
        
        axes[0, 0].bar(range(len(mech_names)), jaccard_scores, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Jaccard Index por Mecanismo')
        axes[0, 0].set_xlabel('Mecanismos')
        axes[0, 0].set_ylabel('Jaccard Index')
        axes[0, 0].set_xticks(range(len(mech_names)))
        axes[0, 0].set_xticklabels([f'M{i+1}' for i in range(len(mech_names))], rotation=45)
        
        axes[0, 1].bar(range(len(mech_names)), f1_scores, alpha=0.7, color='lightcoral')
        axes[0, 1].set_title('F1 Score por Mecanismo')
        axes[0, 1].set_xlabel('Mecanismos')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_xticks(range(len(mech_names)))
        axes[0, 1].set_xticklabels([f'M{i+1}' for i in range(len(mech_names))], rotation=45)
        
        # 2. Distribución de tamaños de mecanismos
        gt_sizes = [mech_metrics[name]['gt_genes'] for name in mech_names]
        disc_sizes = [mech_metrics[name]['disc_genes'] for name in mech_names]
        
        x = np.arange(len(mech_names))
        width = 0.35
        
        axes[0, 2].bar(x - width/2, gt_sizes, width, label='Ground Truth', alpha=0.7, color='green')
        axes[0, 2].bar(x + width/2, disc_sizes, width, label='Descubierto', alpha=0.7, color='orange')
        axes[0, 2].set_title('Tamaño de Mecanismos')
        axes[0, 2].set_xlabel('Mecanismos')
        axes[0, 2].set_ylabel('Número de Genes')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels([f'M{i+1}' for i in range(len(mech_names))])
        axes[0, 2].legend()
        
        # 3. Métricas de asignación
        assign_metrics = assignment_results['assignment_metrics']
        if assign_metrics:
            assign_names = list(assign_metrics.keys())
            accuracies = [assign_metrics[name]['accuracy'] for name in assign_names]
            assign_f1s = [assign_metrics[name]['f1'] for name in assign_names]
            
            axes[1, 0].bar(range(len(assign_names)), accuracies, alpha=0.7, color='gold')
            axes[1, 0].set_title('Accuracy de Asignaciones')
            axes[1, 0].set_xlabel('Mecanismos')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].set_xticks(range(len(assign_names)))
            axes[1, 0].set_xticklabels([f'M{i+1}' for i in range(len(assign_names))], rotation=45)
            
            axes[1, 1].bar(range(len(assign_names)), assign_f1s, alpha=0.7, color='mediumpurple')
            axes[1, 1].set_title('F1 Score de Asignaciones')
            axes[1, 1].set_xlabel('Mecanismos')
            axes[1, 1].set_ylabel('F1 Score')
            axes[1, 1].set_xticks(range(len(assign_names)))
            axes[1, 1].set_xticklabels([f'M{i+1}' for i in range(len(assign_names))], rotation=45)
        
        # 4. Resumen de métricas globales
        global_metrics = [
            mechanism_results['global_metrics']['mean_jaccard'],
            mechanism_results['global_metrics']['mean_f1'],
            assignment_results['global_assignment_metrics']['mean_accuracy'],
            assignment_results['global_assignment_metrics']['mean_f1'],
            assignment_results['global_assignment_metrics']['adjusted_rand_index']
        ]
        
        metric_names = ['Jaccard\n(Mech)', 'F1\n(Mech)', 'Accuracy\n(Assign)', 'F1\n(Assign)', 'ARI']
        
        bars = axes[1, 2].bar(range(len(global_metrics)), global_metrics, 
                             alpha=0.7, color=['skyblue', 'lightcoral', 'gold', 'mediumpurple', 'lightgreen'])
        axes[1, 2].set_title('Métricas Globales')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_xticks(range(len(metric_names)))
        axes[1, 2].set_xticklabels(metric_names)
        axes[1, 2].set_ylim(0, 1)
        
        # Añadir valores en las barras
        for bar, value in zip(bars, global_metrics):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
            print("   💾 Gráficos guardados en 'evaluation_results.png'")
        
        plt.show()
        
        return fig

def run_evaluation(discovered_mechanisms, discovered_assignments, M_ground_truth, A_ground_truth, 
                  algorithm_name="Algorithm", execution_time=None, memory_usage=None):
    """
    Función principal para ejecutar evaluación completa
    """
    print("🎯 Iniciando evaluación completa...")
    print("="*60)
    
    evaluator = MechanismEvaluator()
    evaluator.load_ground_truth(M_ground_truth, A_ground_truth)
    
    # 1. Evaluar descubrimiento de mecanismos
    mechanism_results = evaluator.evaluate_mechanism_discovery(discovered_mechanisms, M_ground_truth)
    
    # 2. Evaluar asignaciones de muestras
    assignment_results = evaluator.evaluate_sample_assignments(discovered_assignments, A_ground_truth)
    
    # 3. Evaluar eficiencia computacional (si se proporciona)
    computational_results = None
    if execution_time is not None:
        computational_results = evaluator.compute_computational_metrics(
            algorithm_name, execution_time, memory_usage
        )
    
    # 4. Generar reporte
    report = evaluator.generate_evaluation_report(
        mechanism_results, assignment_results, computational_results
    )
    
    # 5. Visualizar resultados
    evaluator.visualize_results(mechanism_results, assignment_results)
    
    print("="*60)
    print("✅ Evaluación completada!")
    
    return report

if __name__ == "__main__":
    # Test de evaluación
    print("🧪 Probando evaluador...")
    
    try:
        # Cargar ground truth
        with open('synthetic_data/M_ground_truth.json', 'r') as f:
            M_ground_truth = json.load(f)
        A_ground_truth = np.load('synthetic_data/A_ground_truth.npy')
        
        # Simular resultados descubiertos (para testing)
        discovered_mechanisms = {
            'mechanism_0': {
                'driver_vars': [10, 25, 47, 89],
                'effects': [2.0, 1.5, -1.0, 2.5],
                'type': 'disease'
            },
            'mechanism_1': {
                'driver_vars': [200, 301],
                'effects': [1.2, -0.5],
                'type': 'disease'
            }
        }
        
        discovered_assignments = np.random.randint(0, 2, A_ground_truth.shape)
        
        # Ejecutar evaluación
        report = run_evaluation(
            discovered_mechanisms, 
            discovered_assignments, 
            M_ground_truth, 
            A_ground_truth,
            algorithm_name="Test Algorithm",
            execution_time=45.2
        )
        
        print(f"\n🏆 Score final: {report['global_score']:.2f}/100")
        
    except FileNotFoundError:
        print("❌ No se encontraron datos de ground truth. Ejecuta primero el generador.")