import numpy as np
from scipy.stats import nbinom
import json

def generate_synthetic_scrna_data(num_samples, num_vars, mechanism_defs, redundancy_level):
    """
    Generador de datos sintéticos de scRNA-seq para el hackathon BioDiscovery

    Parameters:
    -----------
    num_samples : int
        Número de muestras/pacientes
    num_vars : int  
        Número total de variables/genes
    mechanism_defs : list of dict
        Lista de mecanismos. Cada dict debe tener:
        - 'vars': list de índices de variables conductoras
        - 'effects': list de efectos (+ para sobre-expresión, - para inhibición)
        - 'type': 'disease' o 'common'
    redundancy_level : float
        Nivel de redundancia (0-1). Proporción de variables redundantes por mecanismo

    Returns:
    --------
    dict con:
    - X: matriz de datos (num_samples x num_vars)
    - Y: etiquetas (0=control, 1=enfermo)
    - M_ground_truth: composición real de mecanismos
    - A_ground_truth: matriz de asignación paciente-mecanismo
    """

    # Inicializar matrices
    X = np.zeros((num_samples, num_vars))
    Y = np.zeros(num_samples, dtype=int)

    # Matriz de asignación: pacientes x mecanismos
    num_mechanisms = len(mechanism_defs)
    A_ground_truth = np.zeros((num_samples, num_mechanisms))

    # Diccionario para tracking de mecanismos
    M_ground_truth = {}

    # Parámetros base para simulación realista
    base_expression = 2.0  # expresión basal
    noise_level = 0.5
    dropout_rate = 0.3  # 30% de dropout típico en scRNA-seq

    # Paso 1: Generar expresión basal para todos los genes
    for i in range(num_samples):
        for j in range(num_vars):
            # Expresión basal con distribución binomial negativa
            mu = base_expression
            r = 5  # parámetro de dispersión
            p = r / (r + mu)
            X[i, j] = nbinom.rvs(r, p)

    # Paso 2: Procesar cada mecanismo
    for mech_idx, mechanism in enumerate(mechanism_defs):
        vars_conductoras = mechanism['vars']
        effects = mechanism['effects']
        mech_type = mechanism['type']
        
        # 🔹 Redundancia específica por mecanismo
        redundancy = mechanism.get('redundancy', redundancy_level)  
        num_redundant = int(len(vars_conductoras) * redundancy)
        vars_redundantes = []

        if num_redundant > 0:
            used_vars = set(vars_conductoras)
            for prev_mech in M_ground_truth.values():
                used_vars.update(prev_mech.get('driver_vars', []))
                used_vars.update(prev_mech.get('redundant_vars', []))
            
            available_vars = [v for v in range(num_vars) if v not in used_vars]
            if len(available_vars) >= num_redundant:
                vars_redundantes = np.random.choice(
                    available_vars, num_redundant, replace=False
                ).tolist()
        # Guardar composición del mecanismo
        M_ground_truth[f'mechanism_{mech_idx+1}'] = {
            'driver_vars': vars_conductoras,
            'redundant_vars': vars_redundantes,
            'effects': effects,
            'type': mech_type
        }

        # Paso 3: Asignar mecanismos a pacientes
        if mech_type == 'disease':
            prob_activation = 0.7
        else:  # common
            prob_activation = 0.4

        for i in range(num_samples):
            # Decidir si este paciente tiene este mecanismo activo
            should_activate = False

            if mech_type == 'disease':
                if np.random.rand() < prob_activation:
                    should_activate = True
                    Y[i] = 1  # marcar como enfermo
            else:  # common
                if np.random.rand() < prob_activation:
                    should_activate = True

            if should_activate:
                A_ground_truth[i, mech_idx] = 1

                # Aplicar efectos a variables conductoras
                for var_idx, effect in zip(vars_conductoras, effects):
                    if effect > 0:  # sobre-expresión
                        fold_change = 1 + effect
                        X[i, var_idx] *= fold_change
                        X[i, var_idx] += np.random.poisson(abs(effect))
                    else:  # inhibición
                        fold_change = max(0.1, 1 + effect)
                        X[i, var_idx] *= fold_change

                # Aplicar efectos correlacionados a variables redundantes
                for red_idx, var_idx in enumerate(vars_redundantes):
                    if red_idx < len(effects):
                        correlation_strength = np.random.uniform(0.5, 0.8)
                        weak_effect = effects[red_idx] * correlation_strength

                        if weak_effect > 0:
                            fold_change = 1 + weak_effect
                            X[i, var_idx] *= fold_change
                            X[i, var_idx] += np.random.poisson(abs(weak_effect))
                        else:
                            fold_change = max(0.1, 1 + weak_effect)
                            X[i, var_idx] *= fold_change

    # Paso 4: Aplicar dropout realista
    dropout_mask = np.random.rand(num_samples, num_vars) < dropout_rate
    X[dropout_mask] = 0

    # Paso 5: Añadir ruido técnico y biológico
    technical_noise = np.random.gamma(1, noise_level, X.shape)
    X = X + technical_noise

    # Asegurar que todos los valores sean no negativos
    X = np.maximum(X, 0)

    # Redondear a enteros (conteos de reads)
    X = np.round(X).astype(int)

    return {
        'X': X,
        'Y': Y,
        'M_ground_truth': M_ground_truth,
        'A_ground_truth': A_ground_truth
    }
