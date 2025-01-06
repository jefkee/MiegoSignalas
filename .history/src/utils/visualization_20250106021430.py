import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import seaborn as sns

class SleepVisualizer:
    def create_visualizations(self, stages, raw_data):
        """Create visualization data for Plotly"""
        # Skaičiuojame stadijų pasiskirstymą
        unique, counts = np.unique(stages, return_counts=True)
        distribution_values = [0] * 5  # Inicializuojame visoms 5 stadijoms
        for stage, count in zip(unique, counts):
            distribution_values[stage] = count

        # Sukuriame laiko taškus hipnogramai
        time_points = np.arange(len(stages)) * 30  # 30 sekundžių epochos
        time_str = [f"{t//3600:02d}:{(t%3600)//60:02d}" for t in time_points]

        # Skaičiuojame perėjimų matricą
        transitions = np.zeros((5, 5))
        for i in range(len(stages)-1):
            transitions[stages[i], stages[i+1]] += 1

        return {
            'distribution_values': distribution_values,
            'time_points': time_str,
            'stages': stages.tolist(),
            'transitions': transitions.tolist()
        }