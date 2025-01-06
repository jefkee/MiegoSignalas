import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import io
import base64

class ReportGenerator:
    """Generates sleep analysis reports and visualizations"""
    
    def __init__(self):
        self.stage_names = {
            0: 'Wake',
            1: 'N1',
            2: 'N2', 
            3: 'N3',
            4: 'REM'
        }
        
        # Set style for visualizations
        plt.style.use('seaborn')
        
    def generate_recommendations(self, analysis):
        """Generate recommendations based on sleep analysis"""
        return analysis.get('recommendations', [])

    def create_stage_distribution_plot(self, stages):
        """Create pie chart of sleep stage distribution"""
        plt.figure(figsize=(10, 8))
        
        # Count stages
        unique, counts = np.unique(stages, return_counts=True)
        stage_dist = dict(zip(unique, counts))
        
        # Calculate percentages
        total = sum(counts)
        percentages = {self.stage_names[stage]: (count/total)*100 
                      for stage, count in stage_dist.items()}
        
        # Colors for each stage
        colors = {
            'Wake': '#1f77b4',  # Blue
            'N1': '#ff7f0e',    # Orange
            'N2': '#2ca02c',    # Green
            'N3': '#d62728',    # Red
            'REM': '#9467bd'    # Purple
        }
        
        # Create pie chart
        plt.pie(percentages.values(), 
                labels=[f"{k}\n{v:.1f}%" for k, v in percentages.items()],
                colors=[colors[stage] for stage in percentages.keys()],
                autopct='%1.1f%%',
                startangle=90)
        
        plt.title('Sleep Stage Distribution')
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(image_png).decode()

    def create_hypnogram(self, stages):
        """Create hypnogram visualization"""
        plt.figure(figsize=(15, 5))
        
        # Create time axis (30-second epochs)
        time_hours = np.arange(len(stages)) * 30 / 3600
        
        # Plot stages
        plt.plot(time_hours, stages, 'b-', linewidth=1)
        
        # Customize plot
        plt.ylim(-0.5, 4.5)
        plt.yticks(range(5), ['Wake', 'N1', 'N2', 'N3', 'REM'])
        plt.xlabel('Time (hours)')
        plt.ylabel('Sleep Stage')
        plt.title('Sleep Stages Throughout Night')
        plt.grid(True)
        
        # Add vertical lines every hour
        for hour in range(int(max(time_hours))+1):
            plt.axvline(x=hour, color='gray', linestyle=':', alpha=0.5)
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(image_png).decode()

    def create_transition_matrix(self, stages):
        """Create sleep stage transition visualization"""
        plt.figure(figsize=(10, 8))
        
        # Calculate transition matrix
        transitions = np.zeros((5, 5))
        for i in range(len(stages)-1):
            transitions[stages[i], stages[i+1]] += 1
            
        # Normalize by row
        row_sums = transitions.sum(axis=1, keepdims=True)
        transitions_norm = np.where(row_sums > 0, transitions / row_sums, 0)
        
        # Create heatmap
        sns.heatmap(transitions_norm, 
                   annot=True, 
                   fmt='.2f',
                   xticklabels=self.stage_names.values(),
                   yticklabels=self.stage_names.values(),
                   cmap='YlOrRd')
        
        plt.title('Sleep Stage Transitions')
        plt.xlabel('To Stage')
        plt.ylabel('From Stage')
        
        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(image_png).decode()

    def generate_report(self, analysis):
        """Generate complete sleep analysis report"""
        stages = np.array(analysis.get('stages', []))
        
        return {
            'sleep_quality': {
                'score': analysis.get('quality_score', 0),
                'interpretation': self._get_quality_interpretation(analysis.get('quality_score', 0))
            },
            'sleep_stages': {
                'distribution': self.create_stage_distribution_plot(stages),
                'hypnogram': self.create_hypnogram(stages),
                'transitions': self.create_transition_matrix(stages)
            },
            'timing': analysis.get('timing', {}),
            'recommendations': analysis.get('recommendations', [])
        }
        
    def _get_quality_interpretation(self, score):
        """Get interpretation of sleep quality score"""
        if score >= 90:
            return "Excellent sleep quality"
        elif score >= 75:
            return "Good sleep quality"
        elif score >= 60:
            return "Fair sleep quality"
        else:
            return "Poor sleep quality"