import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from .classifier import SleepClassifier
from src.config.settings import MODEL_PARAMS, REQUIRED_CHANNELS, PREPROCESSING_PARAMS
import os

class SleepAnalyzer:
    """Neural network-based sleep stage analyzer"""
    
    def __init__(self, model_path=None):
        """Initialize the analyzer"""
        self.classifier = SleepClassifier(MODEL_PARAMS)
        if model_path and os.path.exists(model_path):
            self.classifier.model.load_weights(model_path)
            
    def _generate_distribution_chart(self, sleep_quality):
        """Generate sleep stage distribution pie chart"""
        plt.style.use('dark_background')
        plt.figure(figsize=(8, 6))
        # Order stages correctly: Wake, N1, N2, N3, REM
        labels = ['Wake', 'N1', 'N2', 'N3', 'REM']
        sizes = [sleep_quality['distribution'][label] for label in labels]
        colors = ['#ff9999', '#99ff99', '#9999ff', '#ffff99', '#ff99ff']
        
        # Create pie chart with correct stage order
        patches, texts, autotexts = plt.pie(sizes, labels=labels, colors=colors, 
                                          autopct='%1.1f%%', startangle=90)
        
        # Enhance visibility
        plt.setp(autotexts, size=8, weight="bold")
        plt.setp(texts, size=10)
        plt.title('Sleep Stage Distribution')
        plt.axis('equal')
        
        return self._fig_to_base64()
        
    def _generate_hypnogram(self, stages, times):
        """Generate hypnogram plot"""
        plt.style.use('dark_background')
        plt.figure(figsize=(15, 4))
        
        # Map stages to correct order (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
        plt.plot(times/3600, stages, '-', color='blue', linewidth=1)
        plt.yticks([0, 1, 2, 3, 4], ['Wake', 'N1', 'N2', 'N3', 'REM'])
        plt.xlabel('Time (hours)')
        plt.ylabel('Sleep Stage')
        plt.title('Sleep Stages Throughout Night')
        plt.grid(True, alpha=0.3)
        
        # Add time ticks every hour
        max_hours = int(np.ceil(max(times)/3600))
        plt.xticks(range(max_hours + 1), 
                  [f'{h:02d}:00' for h in range(max_hours + 1)])
        
        return self._fig_to_base64()
        
    def _generate_transitions_chart(self, stages):
        """Generate sleep stage transitions heatmap"""
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 8))
        
        # Calculate transitions with correct stage mapping
        transitions = np.zeros((5, 5))
        for i in range(len(stages)-1):
            transitions[stages[i], stages[i+1]] += 1
            
        # Create heatmap with correct stage order
        sns.heatmap(transitions, annot=True, fmt='g', cmap='YlOrRd',
                   xticklabels=['Wake', 'N1', 'N2', 'N3', 'REM'],
                   yticklabels=['Wake', 'N1', 'N2', 'N3', 'REM'],
                   cbar_kws={'label': 'Number of Transitions'})
        plt.title('Sleep Stage Transitions')
        plt.xlabel('To Stage')
        plt.ylabel('From Stage')
        
        return self._fig_to_base64()
        
    def _generate_hourly_distribution(self, stages, times):
        """Generate hourly sleep stage distribution"""
        plt.style.use('dark_background')
        plt.figure(figsize=(15, 5))
        
        # Calculate hourly distribution with correct stage mapping
        hours = times/3600
        max_hour = int(np.ceil(max(hours)))
        hourly_dist = np.zeros((5, max_hour))
        
        for stage, hour in zip(stages, hours):
            hour_idx = int(hour)
            if hour_idx < max_hour:
                hourly_dist[stage, hour_idx] += 1
                
        # Convert counts to percentages
        total_per_hour = np.sum(hourly_dist, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            hourly_dist = np.where(total_per_hour > 0,
                                 (hourly_dist / total_per_hour.reshape(1, -1)) * 100,
                                 0)
        
        # Plot stacked bar chart with correct stage order
        bottom = np.zeros(max_hour)
        labels = ['Wake', 'N1', 'N2', 'N3', 'REM']
        colors = ['#ff9999', '#99ff99', '#9999ff', '#ffff99', '#ff99ff']
        
        for i in range(5):
            plt.bar(range(max_hour), hourly_dist[i], bottom=bottom, 
                   label=labels[i], color=colors[i])
            bottom += hourly_dist[i]
            
        plt.xlabel('Time')
        plt.ylabel('Percentage')
        plt.title('Hourly Sleep Stage Distribution')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylim(0, 100)
        
        # Add time ticks
        plt.xticks(range(max_hour), 
                  [f'{h:02d}:00' for h in range(max_hour)])
        
        return self._fig_to_base64()
        
    def _generate_spectral_density(self, data, sampling_rate):
        """Generate spectral density plot"""
        plt.style.use('dark_background')
        plt.figure(figsize=(12, 6))
        
        # Calculate PSD for each channel
        colors = ['blue', 'red', 'green', 'orange']
        channel_psds = []
        channel_freqs = []
        
        for channel_data in data:
            # Calculate PSD using Welch's method
            freqs, psd = plt.psd(channel_data, Fs=sampling_rate, NFFT=1024)
            plt.clf()  # Clear the automatic plot
            channel_psds.append(psd)
            channel_freqs.append(freqs)
        
        # Create new figure for custom plot
        plt.figure(figsize=(12, 6))
        
        # Plot each channel's PSD
        for i, (freqs, psd) in enumerate(zip(channel_freqs, channel_psds)):
            plt.semilogy(freqs, psd, color=colors[i], label=f'Channel {i+1}', alpha=0.7)
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (µV²/Hz)')
        plt.title('Power Spectral Density')
        plt.legend(loc='upper right')
        
        # Add frequency band annotations
        bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30)
        }
        
        band_colors = ['#ffcccc', '#ccffcc', '#cccfff', '#ffffcc']
        max_psd = max(np.max(psd) for psd in channel_psds)
        min_psd = max(min(np.min(psd[psd > 0]) for psd in channel_psds), 1e-6)  # Avoid zero
        
        # Add colored bands
        for (band, (fmin, fmax)), color in zip(bands.items(), band_colors):
            plt.axvspan(fmin, fmax, color=color, alpha=0.2)
            plt.text(np.mean([fmin, fmax]), max_psd * 1.1, band, 
                    horizontalalignment='center', verticalalignment='bottom')
        
        plt.xlim(0, 50)  # Limit to relevant frequency range
        plt.ylim(min_psd * 0.1, max_psd * 10)  # Adjust y-axis to show labels
        
        return self._fig_to_base64()
        
    def _fig_to_base64(self):
        """Convert matplotlib figure to base64 string"""
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
        
    def analyze_recording(self, edf_path):
        """Analyze sleep stages from an EDF recording"""
        default_response = {
            'stages': [],
            'times': [],
            'confidence': [],
            'sleep_quality': {
                'score': 0,
                'interpretation': 'No data available for analysis',
                'distribution': {
                    'Wake': 0,
                    'N1': 0,
                    'N2': 0,
                    'N3': 0,
                    'REM': 0
                }
            },
            'recording_info': {
                'start_time': None,
                'duration': 0,
                'sleep_time': 0,
                'sleep_onset': 0,
                'wake_time': 0,
                'efficiency': 0
            },
            'visualizations': {
                'distribution': None,
                'hypnogram': None,
                'transitions': None,
                'hourly_distribution': None,
                'spectral_density': None
            },
            'recommendations': [
                "Unable to analyze sleep data. Please ensure the file format is correct and contains required channels."
            ]
        }
        
        try:
            # Load EDF file
            raw = mne.io.read_raw_edf(edf_path, preload=True)
            raw.pick_channels(REQUIRED_CHANNELS)
            raw.filter(l_freq=PREPROCESSING_PARAMS['highpass'],
                      h_freq=PREPROCESSING_PARAMS['lowpass'])
            
            data = raw.get_data()
            window_size = PREPROCESSING_PARAMS['window_size'] * PREPROCESSING_PARAMS['sampling_rate']
            overlap = int(window_size * PREPROCESSING_PARAMS['overlap'])
            
            segments = []
            for start in range(0, data.shape[1] - window_size + 1, window_size - overlap):
                segment = data[:, start:start + window_size]
                if segment.shape[1] == window_size:
                    segments.append(segment.T)
                    
            segments = np.array(segments)
            
            if len(segments) == 0:
                return default_response
            
            predictions = self.analyze(segments)
            stages = np.argmax(predictions, axis=1)
            times = np.arange(0, len(stages)) * (window_size - overlap) / PREPROCESSING_PARAMS['sampling_rate']
            
            sleep_quality = self._calculate_sleep_quality(stages)
            recommendations = self._generate_recommendations(sleep_quality)
            
            # Calculate recording information
            recording_info = self._calculate_recording_info(raw, stages, times)
            
            # Generate visualizations
            visualizations = {
                'distribution': self._generate_distribution_chart(sleep_quality),
                'hypnogram': self._generate_hypnogram(stages, times),
                'transitions': self._generate_transitions_chart(stages),
                'hourly_distribution': self._generate_hourly_distribution(stages, times),
                'spectral_density': self._generate_spectral_density(data, PREPROCESSING_PARAMS['sampling_rate'])
            }
            
            return {
                'stages': stages.tolist(),
                'times': times.tolist(),
                'confidence': np.max(predictions, axis=1).tolist(),
                'sleep_quality': sleep_quality,
                'recording_info': recording_info,
                'visualizations': visualizations,
                'recommendations': recommendations
            }
            
        except Exception as e:
            print(f"Error analyzing recording: {str(e)}")
            return default_response
        
    def analyze(self, eeg_data):
        """Analyze sleep stages from preprocessed EEG data"""
        # Make predictions
        predictions = self.classifier.predict(eeg_data)
        
        # Ensure correct stage mapping (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
        return predictions
        
    def _calculate_sleep_quality(self, stages):
        """Calculate sleep quality metrics"""
        total_epochs = len(stages)
        if total_epochs == 0:
            return {
                'score': 0,
                'interpretation': 'No data available for analysis',
                'distribution': {
                    'Wake': 0,
                    'N1': 0,
                    'N2': 0,
                    'N3': 0,
                    'REM': 0
                }
            }
            
        # Count stages (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)
        stage_counts = np.bincount(stages, minlength=5)
        stage_percentages = (stage_counts / total_epochs) * 100
        
        # Calculate quality score with appropriate weights
        # Higher weights for N3 (deep sleep) and REM, lower for wake and N1
        weights = [0.2, 0.4, 0.6, 1.0, 0.8]  # Wake, N1, N2, N3, REM
        quality_score = 0
        
        # Target percentages for optimal sleep
        targets = {
            'Wake': 5,    # 5% wake is normal
            'N1': 5,      # 5% N1 is normal
            'N2': 45,     # 45-55% N2 is normal
            'N3': 25,     # 20-25% N3 is optimal
            'REM': 20     # 20-25% REM is optimal
        }
        
        # Calculate score based on how close to optimal percentages
        for i, (stage, target) in enumerate(targets.items()):
            actual = stage_percentages[i]
            if stage in ['Wake', 'N1']:
                # Penalize excess wake/N1 time
                quality_score += weights[i] * (100 - max(0, actual - target)) / 100
            else:
                # Reward getting close to target for other stages
                quality_score += weights[i] * (100 - min(abs(actual - target), 100)) / 100
        
        # Normalize score to 0-100 range
        max_possible_score = sum(weights)
        score = min(100, int((quality_score / max_possible_score) * 100))
        
        # Generate interpretation
        interpretation = self._get_sleep_quality_interpretation(score, stage_percentages)
        
        # Create stage distribution with correct mapping
        stage_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
        distribution = {
            name: float(count) / total_epochs * 100 
            for name, count in zip(stage_names, stage_counts)
        }
        
        return {
            'score': score,
            'interpretation': interpretation,
            'distribution': distribution
        }

    def _get_sleep_quality_interpretation(self, score, stage_percentages):
        """Generate sleep quality interpretation based on score and stage percentages"""
        if score >= 90:
            return "Excellent sleep quality - Your sleep architecture shows optimal patterns"
        elif score >= 80:
            return "Good sleep quality - Your sleep structure is generally healthy"
        elif score >= 70:
            return "Fair sleep quality - Some aspects of your sleep could be improved"
        elif score >= 60:
            return "Poor sleep quality - Consider implementing sleep hygiene improvements"
        else:
            return "Very poor sleep quality - Consultation with a sleep specialist is recommended"
        
    def _generate_recommendations(self, sleep_quality):
        """Generate sleep recommendations based on analysis"""
        recommendations = []
        dist = sleep_quality['distribution']
        
        # Wake recommendations
        if dist['Wake'] > 20:
            recommendations.extend([
                "High wake time detected - Consider these improvements:",
                "- Evaluate your sleep environment for disturbances (noise, light, temperature)",
                "- Establish a consistent bedtime routine",
                "- Avoid using electronic devices before bed",
                "- Consider using white noise or earplugs if noise is an issue"
            ])
        
        # Light sleep (N1) recommendations
        if dist['N1'] > 15:
            recommendations.extend([
                "Higher than optimal light sleep detected:",
                "- Reduce caffeine intake, especially in the afternoon",
                "- Practice stress-reduction techniques before bed",
                "- Consider using blackout curtains to minimize light disturbance",
                "- Maintain a consistent sleep schedule"
            ])
        
        # N2 sleep recommendations
        if dist['N2'] < 45:
            recommendations.extend([
                "Lower than optimal N2 sleep detected:",
                "- Establish a regular exercise routine (but not too close to bedtime)",
                "- Create a relaxing pre-sleep routine",
                "- Consider meditation or deep breathing exercises"
            ])
        
        # Deep sleep (N3) recommendations
        if dist['N3'] < 15:
            recommendations.extend([
                "Low deep sleep detected:",
                "- Exercise regularly during the day",
                "- Keep your bedroom cool (around 18-20°C)",
                "- Avoid alcohol before bedtime",
                "- Consider timing your sleep with your natural circadian rhythm"
            ])
        
        # REM sleep recommendations
        if dist['REM'] < 20:
            recommendations.extend([
                "Low REM sleep detected:",
                "- Reduce stress through relaxation techniques",
                "- Maintain a consistent sleep schedule",
                "- Avoid alcohol and caffeine before bed",
                "- Practice good sleep hygiene"
            ])
        
        # Sleep quality score based recommendations
        score = sleep_quality['score']
        if score < 60:
            recommendations.extend([
                "Critical sleep quality improvements needed:",
                "- Consider consulting a sleep specialist",
                "- Keep a detailed sleep diary",
                "- Evaluate any medications that might affect sleep",
                "- Check for underlying health conditions"
            ])
        elif score < 75:
            recommendations.extend([
                "Sleep quality needs improvement:",
                "- Improve sleep hygiene practices",
                "- Create a bedtime routine",
                "- Limit screen time before bed",
                "- Ensure your mattress and pillows are comfortable"
            ])
        
        # Add general recommendations if sleep is generally good
        if score >= 80 and len(recommendations) == 0:
            recommendations.extend([
                "Your sleep quality is good. To maintain it:",
                "- Continue your current sleep routine",
                "- Monitor your sleep patterns regularly",
                "- Make adjustments if you notice changes",
                "- Practice preventive sleep hygiene"
            ])
        
        # Always add these basic recommendations
        recommendations.extend([
            "",
            "General sleep hygiene tips:",
            "- Maintain a consistent sleep schedule",
            "- Keep your bedroom dark, quiet, and cool",
            "- Avoid large meals close to bedtime",
            "- Limit exposure to blue light before sleep",
            "- Exercise regularly, but not too close to bedtime"
        ])
        
        return recommendations

    def _calculate_recording_info(self, raw, stages, times):
        """Calculate recording information metrics"""
        # Get recording start time
        start_time = raw.info['meas_date']
        if start_time is None:
            start_time = "Not available"
        
        # Calculate durations
        total_duration = len(stages) * 30 / 60  # Convert 30-second epochs to minutes
        
        # Calculate sleep time (excluding wake)
        sleep_epochs = np.sum(stages != 0)  # 0 is Wake
        total_sleep_time = sleep_epochs * 30 / 60  # Convert to minutes
        
        # Find sleep onset (first non-wake epoch)
        sleep_onset_epochs = np.where(stages != 0)[0]
        sleep_onset = (sleep_onset_epochs[0] * 30 / 60) if len(sleep_onset_epochs) > 0 else 0
        
        # Find final wake time (last non-wake epoch)
        wake_time = (sleep_onset_epochs[-1] * 30 / 60) if len(sleep_onset_epochs) > 0 else 0
        
        # Calculate sleep efficiency
        sleep_efficiency = (total_sleep_time / total_duration * 100) if total_duration > 0 else 0
        
        return {
            'start_time': start_time,
            'duration': total_duration,
            'sleep_time': total_sleep_time,
            'sleep_onset': sleep_onset,
            'wake_time': wake_time,
            'efficiency': sleep_efficiency
        }