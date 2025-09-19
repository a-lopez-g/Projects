
import numpy as np
import json
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class BioDiscoveryAlgorithm:
    """
    Algorithm for discovering hidden mechanisms in high-dimensional biological data
    Designed for the BioDiscovery Hackathon - Part B: Discovery Algorithm
    """

    def __init__(self, params=None):
        """
        Initialize the discovery algorithm with optional parameters

        Parameters:
        -----------
        params : dict, optional
            Dictionary containing algorithm parameters:
            - n_mechanisms: number of mechanisms to discover (default: auto-detect)
            - feature_selection_k: number of top features to select (default: 1000)
            - clustering_method: 'kmeans' or 'hierarchical' (default: 'kmeans')
            - effect_threshold: threshold for determining variable effects (default: 0.1)
        """
        default_params = {
            'n_mechanisms': None,  # Auto-detect
            'feature_selection_k': 1000,
            'clustering_method': 'kmeans',
            'effect_threshold': 0.1,
            'max_mechanisms': 10,
            'min_mechanism_size': 3  # Reduced from 5 to 3
        }

        if params is None:
            params = {}

        self.params = {**default_params, **params}
        self.discovered_mechanisms = None
        self.discovered_assignments = None

    def discover_mechanisms(self, X, Y):
        """
        Main discovery function - implements the core algorithm

        Parameters:
        -----------
        X : numpy.ndarray
            Data matrix (samples x variables)
        Y : numpy.ndarray
            Labels (0 for control, 1 for disease)

        Returns:
        --------
        tuple : (Discovered_Mechanisms, Discovered_Assignments)
            - Discovered_Mechanisms: dict with mechanism compositions
            - Discovered_Assignments: matrix indicating which mechanisms are active per sample
        """
        print("Starting mechanism discovery...")

        # Step 1: Feature selection to reduce dimensionality
        X_selected, selected_features = self._feature_selection(X, Y)
        print(f"Selected {len(selected_features)} features from {X.shape[1]} total")

        # Step 2: Identify disease-specific patterns
        disease_mask = Y == 1
        X_disease = X_selected[disease_mask]

        if X_disease.shape[0] == 0:
            print("No disease samples found!")
            return {}, np.zeros((X.shape[0], 1))

        # Step 3: Apply dimensionality reduction to find latent mechanisms
        mechanisms_data = self._find_latent_mechanisms(X_disease)

        # Step 4: Cluster samples to identify mechanism assignments
        assignments = self._assign_mechanisms(X_selected, Y, mechanisms_data)

        # Step 5: Characterize discovered mechanisms
        mechanisms = self._characterize_mechanisms(X_selected, Y, assignments, selected_features)

        self.discovered_mechanisms = mechanisms
        self.discovered_assignments = assignments

        print(f"Discovered {len(mechanisms)} mechanisms")
        return mechanisms, assignments

    def _feature_selection(self, X, Y):
        """
        Select most informative features using statistical tests
        """
        # Handle case where we have fewer features than requested
        k = min(self.params['feature_selection_k'], X.shape[1])

        # Use F-test for feature selection
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, Y)
        selected_features = selector.get_support(indices=True)

        return X_selected, selected_features

    def _find_latent_mechanisms(self, X_disease):
        """
        Use ICA to find independent components representing mechanisms
        """
        # Determine number of components - be more conservative
        n_components = min(
            self.params['max_mechanisms'], 
            X_disease.shape[0] - 1,  # Must be less than n_samples
            X_disease.shape[1],      # Must be less than n_features
            5  # Cap at 5 for stability
        )
        if n_components < 1:
            n_components = 1

        # Apply ICA to find independent mechanisms
        try:
            ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
            mechanisms_data = ica.fit_transform(X_disease)
        except Exception as e:
            print(f"ICA failed: {e}, using PCA instead")
            pca = PCA(n_components=n_components, random_state=42)
            mechanisms_data = pca.fit_transform(X_disease)

        return mechanisms_data

    def _assign_mechanisms(self, X, Y, mechanisms_data):
        """
        Assign mechanisms to all samples (including controls)
        """
        disease_mask = Y == 1
        n_samples = X.shape[0]
        n_mechanisms = mechanisms_data.shape[1]

        # Initialize assignment matrix
        assignments = np.zeros((n_samples, n_mechanisms))

        # For disease samples, use clustering on mechanism space
        if np.sum(disease_mask) > 0 and mechanisms_data.shape[0] > 0:
            # Use fewer clusters if we have few samples
            n_clusters = min(n_mechanisms, np.sum(disease_mask), 3)

            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                disease_clusters = kmeans.fit_predict(mechanisms_data)
            else:
                disease_clusters = np.zeros(mechanisms_data.shape[0])

            # Convert clusters to mechanism assignments
            for i, cluster in enumerate(disease_clusters):
                sample_idx = np.where(disease_mask)[0][i]
                # Assign mechanism based on strongest component
                mechanism_strengths = np.abs(mechanisms_data[i])
                # Take top 1-2 mechanisms
                n_top = min(2, len(mechanism_strengths))
                top_mechanisms = np.argsort(mechanism_strengths)[-n_top:]
                assignments[sample_idx, top_mechanisms] = 1

        return assignments

    def _characterize_mechanisms(self, X, Y, assignments, selected_features):
        """
        Characterize each discovered mechanism by identifying key variables
        """
        mechanisms = {}
        n_mechanisms = assignments.shape[1]

        for mech_idx in range(n_mechanisms):
            # Find samples with this mechanism active
            active_samples = assignments[:, mech_idx] == 1

            if np.sum(active_samples) < self.params['min_mechanism_size']:
                continue

            # Compare active vs inactive samples
            X_active = X[active_samples]
            X_inactive = X[~active_samples]

            # Calculate effect sizes for each variable
            effects = []
            significant_vars = []

            for var_idx in range(X.shape[1]):
                if np.sum(active_samples) > 1 and np.sum(~active_samples) > 1:
                    try:
                        # T-test to find significant variables
                        t_stat, p_val = stats.ttest_ind(X_active[:, var_idx], X_inactive[:, var_idx])

                        if p_val < 0.05:  # Significant difference
                            effect_size = np.mean(X_active[:, var_idx]) - np.mean(X_inactive[:, var_idx])
                            if abs(effect_size) > self.params['effect_threshold']:
                                effects.append(effect_size)
                                significant_vars.append(selected_features[var_idx])
                    except:
                        continue  # Skip if t-test fails

            if len(significant_vars) > 0:
                mechanisms[f'mechanism_{mech_idx}'] = {
                    'vars': significant_vars,
                    'effects': effects,
                    'type': 'disease',  # All discovered mechanisms are disease-related
                    'n_samples': np.sum(active_samples)
                }

        return mechanisms

def discovery_algorithm(X, Y, params=None):
    """
    Main function for the discovery algorithm as required by the hackathon

    Parameters:
    -----------
    X : numpy.ndarray
        Data matrix (samples x variables)
    Y : numpy.ndarray  
        Labels (0 for control, 1 for disease)
    params : dict, optional
        Optional parameters for the algorithm

    Returns:
    --------
    tuple : (Discovered_Mechanisms, Discovered_Assignments)
        - Discovered_Mechanisms: dict with mechanism compositions
        - Discovered_Assignments: matrix indicating which mechanisms are active per sample
    """
    algorithm = BioDiscoveryAlgorithm(params)
    return algorithm.discover_mechanisms(X, Y)

def evaluate_discovery(X, Y, M_ground_truth, A_ground_truth, discovered_mechanisms, discovered_assignments):
    """
    Evaluate the discovery algorithm results against ground truth

    Parameters:
    -----------
    X : numpy.ndarray
        Original data matrix
    Y : numpy.ndarray
        Labels
    M_ground_truth : dict
        Ground truth mechanisms with 'vars', 'effects', 'type'
    A_ground_truth : numpy.ndarray
        Ground truth assignment matrix (samples x mechanisms)
    discovered_mechanisms : dict
        Discovered mechanisms from algorithm
    discovered_assignments : numpy.ndarray
        Discovered assignment matrix

    Returns:
    --------
    dict : Evaluation results with metrics and visualizations
    """

    results = {
        'mechanism_metrics': {},
        'assignment_metrics': {},
        'summary': {}
    }

    # --- 1. Mechanism Discovery Evaluation ---
    print("Evaluating mechanism discovery...")
    
    # Extract ground truth mechanism variables (drivers + redundant)
    gt_mechanisms = {}
    for mech_name, mech_data in M_ground_truth.items():
        mech_type = mech_data.get('type', 'disease')
        
        if mech_type == 'disease':
            drivers = mech_data.get('driver_vars', [])
            redundant = mech_data.get('redundant_vars', [])
            all_vars = list(set(drivers + redundant))
            
            gt_mechanisms[mech_name] = set(all_vars)

    # Evaluate each discovered mechanism
    mechanism_jaccards = []
    if len(discovered_mechanisms) == 0:
        print("No mechanisms discovered!")
        mechanism_jaccards = [0]
    else:
        for disc_name, disc_data in discovered_mechanisms.items():
            discovered_vars = set(disc_data['vars'])

            # Find best matching ground truth mechanism
            best_jaccard = 0
            best_match = None

            for gt_name, gt_vars in gt_mechanisms.items():
                if len(discovered_vars | gt_vars) > 0:
                    jaccard = len(discovered_vars & gt_vars) / len(discovered_vars | gt_vars)
                    if jaccard > best_jaccard:
                        best_jaccard = jaccard
                        best_match = gt_name

            # Handle case where no ground truth mechanisms exist
            if best_match is None:
                best_match = "None"
                best_jaccard = 0

            results['mechanism_metrics'][disc_name] = {
                'best_match': best_match,
                'jaccard_score': best_jaccard,
                'precision': len(discovered_vars & gt_mechanisms.get(best_match, set())) / len(discovered_vars) if len(discovered_vars) > 0 else 0,
                'recall': len(discovered_vars & gt_mechanisms.get(best_match, set())) / len(gt_mechanisms.get(best_match, set())) if best_match != "None" and best_match in gt_mechanisms and len(gt_mechanisms[best_match]) > 0 else 0,
                'num_vars': len(discovered_vars)
            }

            mechanism_jaccards.append(best_jaccard)

    # --- 2. Assignment Evaluation ---
    print("Evaluating mechanism assignments...")

    if discovered_assignments.shape[0] == A_ground_truth.shape[0]:
        try:
            # Convert to cluster labels for comparison
            gt_labels = []
            disc_labels = []

            for i in range(A_ground_truth.shape[0]):
                # Ground truth: find active mechanisms
                gt_active = np.where(A_ground_truth[i] == 1)[0]
                gt_label = tuple(sorted(gt_active)) if len(gt_active) > 0 else (-1,)
                gt_labels.append(gt_label)

                # Discovered: find active mechanisms
                disc_active = np.where(discovered_assignments[i] == 1)[0]
                disc_label = tuple(sorted(disc_active)) if len(disc_active) > 0 else (-1,)
                disc_labels.append(disc_label)

            # Convert tuples to integers for sklearn metrics
            unique_gt = list(set(gt_labels))
            unique_disc = list(set(disc_labels))

            gt_numeric = [unique_gt.index(label) for label in gt_labels]
            disc_numeric = [unique_disc.index(label) for label in disc_labels]

            # Calculate metrics
            ari = adjusted_rand_score(gt_numeric, disc_numeric)
            accuracy = accuracy_score(gt_numeric, disc_numeric)

            results['assignment_metrics'] = {
                'adjusted_rand_index': ari,
                'accuracy': accuracy,
                'num_gt_patterns': len(unique_gt),
                'num_discovered_patterns': len(unique_disc)
            }
        except Exception as e:
            print(f"Assignment evaluation failed: {e}")
            results['assignment_metrics'] = {
                'adjusted_rand_index': 0,
                'accuracy': 0,
                'num_gt_patterns': 0,
                'num_discovered_patterns': 0
            }
    else:
        print(f"Assignment shape mismatch: discovered {discovered_assignments.shape} vs ground truth {A_ground_truth.shape}")
        results['assignment_metrics'] = {
            'adjusted_rand_index': 0,
            'accuracy': 0,
            'num_gt_patterns': 0,
            'num_discovered_patterns': 0
        }

    # --- 3. Summary Statistics ---
    results['summary'] = {
        'num_gt_mechanisms': len(gt_mechanisms),
        'num_discovered_mechanisms': len(discovered_mechanisms),
        'avg_mechanism_jaccard': np.mean(mechanism_jaccards) if mechanism_jaccards else 0,
        'max_mechanism_jaccard': np.max(mechanism_jaccards) if mechanism_jaccards else 0,
        'mechanism_recovery_rate': np.sum([j > 0.3 for j in mechanism_jaccards]) / len(gt_mechanisms) if len(gt_mechanisms) > 0 else 0
    }

    # --- 4. Visualizations ---
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Assignment heatmaps comparison
        sns.heatmap(A_ground_truth.T, ax=axes[0,0], cmap='Reds', cbar=True)
        axes[0,0].set_title('Ground Truth Assignments')
        axes[0,0].set_xlabel('Samples')
        axes[0,0].set_ylabel('Mechanisms')
        
        sns.heatmap(discovered_assignments.T, ax=axes[0,1], cmap='Blues', cbar=True)
        axes[0,1].set_title('Discovered Assignments')
        axes[0,1].set_xlabel('Samples')
        axes[0,1].set_ylabel('Mechanisms')
        
        # Plot 2: Mechanism overlap scores
        if mechanism_jaccards and len(mechanism_jaccards) > 0:
            axes[1,0].bar(range(len(mechanism_jaccards)), mechanism_jaccards)
            axes[1,0].set_title('Mechanism Jaccard Scores')
            axes[1,0].set_xlabel('Discovered Mechanisms')
            axes[1,0].set_ylabel('Jaccard Score')
            axes[1,0].axhline(y=0.3, color='r', linestyle='--', label='Good threshold')
            axes[1,0].legend()
        else:
            axes[1,0].text(0.5, 0.5, 'No mechanisms discovered', ha='center', va='center')
            axes[1,0].set_title('Mechanism Jaccard Scores')
        
        # Plot 3: Summary metrics
        metrics_names = ['Avg Jaccard', 'ARI', 'Accuracy', 'Recovery Rate']
        metrics_values = [
            results['summary']['avg_mechanism_jaccard'],
            results['assignment_metrics'].get('adjusted_rand_index', 0),
            results['assignment_metrics'].get('accuracy', 0),
            results['summary']['mechanism_recovery_rate']
        ]
        
        axes[1,1].bar(metrics_names, metrics_values)
        axes[1,1].set_title('Summary Metrics')
        axes[1,1].set_ylabel('Score')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save the figure instead of showing it
        plt.savefig('evaluation_results_ICA.png', dpi=300, bbox_inches='tight')
        print("✅ Saved evaluation plots as 'evaluation_results.png'")
        plt.close()  # Close to free memory
    
    except Exception as e:
        print(f"Visualization failed: {e}")

    # --- 5. Print Results ---
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)

    print(f"\nMECHANISM DISCOVERY:")
    print(f"  Ground Truth Mechanisms: {results['summary']['num_gt_mechanisms']}")
    print(f"  Discovered Mechanisms: {results['summary']['num_discovered_mechanisms']}")
    print(f"  Average Jaccard Score: {results['summary']['avg_mechanism_jaccard']:.3f}")
    print(f"  Max Jaccard Score: {results['summary']['max_mechanism_jaccard']:.3f}")
    print(f"  Recovery Rate (>0.3): {results['summary']['mechanism_recovery_rate']:.3f}")

    if 'adjusted_rand_index' in results['assignment_metrics']:
        print(f"\nASSIGNMENT ACCURACY:")
        print(f"  Adjusted Rand Index: {results['assignment_metrics']['adjusted_rand_index']:.3f}")
        print(f"  Accuracy: {results['assignment_metrics']['accuracy']:.3f}")

    print("\nDETAILED MECHANISM MATCHES:")
    for mech_name, metrics in results['mechanism_metrics'].items():
        print(f"  {mech_name} -> {metrics['best_match']} (Jaccard: {metrics['jaccard_score']:.3f})")

    return results


# Updated main function to include evaluation
def run_full_evaluation(X, Y, M_ground_truth, A_ground_truth, params=None):
    """
    Complete pipeline: discover mechanisms and evaluate results

    Parameters:
    -----------
    X, Y : data and labels
    M_ground_truth, A_ground_truth : ground truth from data generator
    params : algorithm parameters

    Returns:
    --------
    tuple : (discovered_mechanisms, discovered_assignments, evaluation_results)
    """
    # Run discovery
    discovered_mechanisms, discovered_assignments = discovery_algorithm(X, Y, params)

    # Evaluate results
    evaluation_results = evaluate_discovery(
        X, Y, M_ground_truth, A_ground_truth, 
        discovered_mechanisms, discovered_assignments
    )

    return discovered_mechanisms, discovered_assignments, evaluation_results

# Example usage and testing
if __name__ == "__main__":
    # Test with synthetic data

    X = np.load('synthetic_data/first_try/X_matrix.npy')
    Y = np.load('synthetic_data/first_try/Y_labels.npy')
    A = np.load("synthetic_data/first_try/A_ground_truth.npy")
    with open('synthetic_data/first_try/M_ground_truth.json') as f:
        M = json.load(f)

    print(f"Data loaded: X shape {X.shape}, Y shape {Y.shape}, A shape {A.shape}")
    print(f"Ground truth mechanisms: {list(M.keys())}")

    mechanisms, assignments, eval_results = run_full_evaluation(X, Y, M, A)
