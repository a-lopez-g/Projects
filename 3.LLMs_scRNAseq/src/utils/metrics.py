from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

def compute_prediction_metrics(real_values, predicted_values):
    metrics = {
        "accuracy": accuracy_score(real_values, predicted_values),
        "precision": precision_score(real_values, predicted_values, average='weighted'),
        "recall": recall_score(real_values, predicted_values, average='weighted'),
        "f1_score": f1_score(real_values, predicted_values, average='weighted')
    }
    return metrics

def graph_times(inference_times):
    average_inference_time = np.mean(inference_times)
    print(f"Tiempo promedio de inferencia por instancia: {average_inference_time:.4f} segundos")
    # Graficar la evolución de los tiempos de inferencia
    plt.figure(figsize=(10, 6))
    plt.plot(inference_times, label='Tiempos de inferencia por instancia', marker='o')
    plt.axhline(y=average_inference_time, color='r', linestyle='--', label=f'Tiempo medio = {average_inference_time:.4f}s')
    plt.xlabel('Instancia')
    plt.ylabel('Tiempo de Inferencia (segundos)')
    plt.title('Evolución de los tiempos de inferencia por instancia')
    plt.legend()
    plt.show()
