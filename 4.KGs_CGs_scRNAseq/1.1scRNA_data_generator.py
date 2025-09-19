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
    base_expression = 2.0  # expresión basal, media
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
        
        # Redundancia específica por mecanismo
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

# Ejemplo de uso
if __name__ == "__main__":
    # Definir mecanismos de ejemplo
    mechanism_defs = [
        {
            'vars': [10,25,47,89,156],
            'effects': [2.5,1.8,-1.2,3.0,1.5],
            'type': 'disease',
            'redundancy': 0.5   # 50% redundancia
        },
        {
            'vars': [200,301,445],
            'effects': [1.5,-0.8,2.2],
            'type': 'disease',
            'redundancy': 0.0   # sin redundancia
        },
        {
            'vars': [50,75,100,125],
            'effects': [0.5,0.3,-0.2,0.4],
            'type': 'common',
            'redundancy': 0.2   # 20% redundancia
        }
    ]

    # Generar datos
    data = generate_synthetic_scrna_data(
        num_samples=1000,
        num_vars=500,
        mechanism_defs=mechanism_defs,
        redundancy_level=0.3 # usado solo como default si no se pasa en mechanism_defs
    )

    print(f"Datos generados exitosamente!")
    print(f"X shape: {data['X'].shape}")
    print(f"Y shape: {data['Y'].shape}")
    print(f"Mecanismos: {len(data['M_ground_truth'])}")
    # Save data
    # Guardar X (matriz de expresión)
    np.save('synthetic_data/X_matrix.npy', data['X'])
    # Guardar Y (etiquetas de pacientes)
    np.save('synthetic_data/Y_labels.npy', data['Y'])
    # Guardar M_ground_truth (definición de mecanismos)
    with open('synthetic_data/M_ground_truth.json', 'w') as f:
        json.dump(data['M_ground_truth'], f, indent=4)
    # Guardar A_ground_truth (asignación de mecanismos a pacientes)
    np.save('synthetic_data/A_ground_truth.npy', data['A_ground_truth'])
    print("\n¡Todos los datos generados han sido guardados exitosamente!")


# Más escenarios
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
