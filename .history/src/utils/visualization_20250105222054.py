import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
import seaborn as sns

class SleepVisualizer:
    def __init__(self):
        self.stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
        self.stage_colors = ['#FF4B4B', '#96C37D', '#2D8BC7', '#224F7C', '#45B7D1']
        
    def create_visualizations(self, stages, raw_data):
        return {
            'distribution': self.create_stage_distribution(stages),
            'hypnogram': self.create_hypnogram(stages),
            'transitions': self.create_transition_matrix(stages),
            'hourly_distribution': self.create_hourly_distribution(stages),
            'spectral_density': self.create_spectral_density(raw_data),
            'sleep_cycles': self.create_sleep_cycles(stages)
        }
        
    def create_stage_distribution(self, stages):
        """Create pie chart of sleep stage distribution"""
        if not isinstance(stages, (list, np.ndarray)):
            return ""
            
        # Convert stages to counts
        stage_counts = {}
        for i, name in enumerate(self.stage_names):
            count = sum(1 for s in stages if s == i)
            if count > 0:  # Only include non-zero stages
                stage_counts[name] = count
        
        # Create pie chart
        plt.figure(figsize=(8, 6))
        plt.pie(stage_counts.values(), labels=stage_counts.keys(), autopct='%1.1f%%')
        plt.title('Sleep Stage Distribution')
        
        # Convert plot to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return f'data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}'
        
    def create_hypnogram(self, stages):
        """Create hypnogram visualization"""
        if not isinstance(stages, (list, np.ndarray)) or len(stages) == 0:
            return ""
            
        # Convert epoch indices to hours:minutes
        epoch_duration = 30  # seconds
        total_seconds = np.arange(len(stages)) * epoch_duration
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        
        # Create time labels for x-axis
        time_labels = []
        time_positions = []
        for i in range(0, len(stages), 120):  # Every hour (120 epochs)
            h = int(hours[i])
            m = int(minutes[i])
            time_labels.append(f"{h:02d}:{m:02d}")
            time_positions.append(i)
        
        plt.figure(figsize=(12, 4))
        plt.plot(stages, 'b-', linewidth=1)
        plt.yticks(range(5), self.stage_names)
        plt.xticks(time_positions, time_labels, rotation=45)
        plt.title('Sleep Stages Throughout Night')
        plt.xlabel('Time (HH:MM)')
        plt.ylabel('Sleep Stage')
        plt.grid(True)
        
        return self._fig_to_base64()
        
    def create_transition_matrix(self, stages):
        """Create sleep stage transition visualization"""
        transitions = np.zeros((5, 5))
        for i in range(len(stages)-1):
            transitions[stages[i], stages[i+1]] += 1
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(transitions, annot=True, fmt='g', 
                   xticklabels=self.stage_names, 
                   yticklabels=self.stage_names)
        plt.title('Sleep Stage Transitions')
        plt.xlabel('To Stage')
        plt.ylabel('From Stage')
        
        return self._fig_to_base64()
        
    def create_hourly_distribution(self, stages):
        """Create hourly sleep stage distribution"""
        epochs_per_hour = 120  # 30-second epochs
        hours = len(stages) // epochs_per_hour
        hourly_dist = []
        
        for hour in range(hours):
            start = hour * epochs_per_hour
            end = start + epochs_per_hour
            hour_stages = stages[start:end]
            dist = [sum(s == i for s in hour_stages) for i in range(5)]
            hourly_dist.append([d/epochs_per_hour*100 for d in dist])
        
        plt.figure(figsize=(12, 6))
        bottom = np.zeros(hours)
        
        # Create hour labels
        hour_labels = [f"{i:02d}:00" for i in range(hours)]
        
        for i in range(5):
            values = [dist[i] for dist in hourly_dist]
            plt.bar(range(hours), values, bottom=bottom, 
                   label=self.stage_names[i], color=self.stage_colors[i])
            bottom += values
        
        plt.title('Hourly Sleep Stage Distribution')
        plt.xlabel('Time')
        plt.ylabel('Percentage')
        plt.xticks(range(hours), hour_labels, rotation=45)
        plt.legend()
        
        return self._fig_to_base64()
        
    def create_sleep_cycles(self, stages):
        """Visualize sleep cycles"""
        # Identify NREM-REM cycles
        cycles = []
        current_cycle = []
        in_nrem = False
        
        for stage in stages:
            if not in_nrem and stage in [1, 2, 3]:  # Start NREM
                in_nrem = True
                current_cycle = [stage]
            elif in_nrem:
                if stage == 4:  # REM
                    current_cycle.append(stage)
                    cycles.append(current_cycle)
                    current_cycle = []
                    in_nrem = False
                else:
                    current_cycle.append(stage)
        
        plt.figure(figsize=(12, 6))
        cycle_boundaries = [0]
        for cycle in cycles:
            cycle_boundaries.append(cycle_boundaries[-1] + len(cycle))
        
        plt.plot(stages, 'b-', linewidth=1)
        for boundary in cycle_boundaries[1:-1]:
            plt.axvline(x=boundary, color='r', linestyle='--', alpha=0.5)
        
        plt.yticks(range(5), self.stage_names)
        plt.title('Sleep Cycles')
        plt.xlabel('Time (epochs)')
        plt.ylabel('Sleep Stage')
        
        return self._fig_to_base64()
        
    def create_spectral_density(self, raw_data):
        """Create spectral density plot"""
        from scipy import signal
        
        plt.figure(figsize=(12, 6))
        for i in range(min(3, raw_data.shape[0])):  # Plot first 3 channels
            f, Pxx = signal.welch(raw_data[i], fs=100, nperseg=1024)
            plt.semilogy(f, Pxx)
        
        plt.title('Power Spectral Density')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power Spectral Density')
        plt.grid(True)
        plt.legend(['Channel 1', 'Channel 2', 'Channel 3'])
        
        return self._fig_to_base64()
        
    def _fig_to_base64(self):
        """Convert current figure to base64 string"""
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return f'data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}'