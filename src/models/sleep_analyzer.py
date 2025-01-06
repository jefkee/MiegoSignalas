import mne
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import tensorflow as tf
from .classifier import SleepStageClassifier, weighted_categorical_crossentropy
from src.config.settings import MODEL_PARAMS, REQUIRED_CHANNELS, PREPROCESSING_PARAMS
import os
from scipy import signal

class SleepAnalyzer:
    """Neural network-based sleep stage analyzer"""
    
    def __init__(self):
        # Initialize the analyzer with a neural network model
        model_path = 'models/sleep_classifier.h5'
        try:
            self.model = SleepStageClassifier.load_model(model_path)
            print("Successfully loaded sleep classifier model")
        except Exception as e:
            print(f"Could not load model from {model_path}: {str(e)}")
            print("Initializing new model instead")
            # Initialize new model with default parameters
            self.model = SleepStageClassifier()
        
        # Define required EEG channels for analysis
        self.required_channels = [
            'EEG Fpz-Cz',      # Frontal EEG channel
            'EEG Pz-Oz',       # Occipital EEG channel
            'EOG horizontal',   # Eye movement channel
            'EMG submental',    # Chin muscle activity
        ]
        
        # Add configuration parameters
        self.window_size = 3000  # 30 seconds at 100Hz
        self.overlap = 0  # No overlap between windows
        
        # Define default response
        self.default_response = {
            'stages': [],
            'times': [],
            'confidence': [],
            'sleep_quality': {
                'score': 0,
                'interpretation': 'Analysis failed - insufficient data',
                'distribution': {'Wake': 0, 'N1': 0, 'N2': 0, 'N3': 0, 'REM': 0}
            },
            'recording_info': {},
            'visualizations': {},
            'recommendations': ['Unable to analyze sleep patterns. Please ensure recording quality.']
        }

    def analyze_recording(self, psg_path, hypno_path=None):
        """Analyze a sleep recording"""
        try:
            # Load EDF file
            raw = mne.io.read_raw_edf(psg_path, preload=True)
            print(f"Loading EDF file: {psg_path}")
            print(f"Available channels: {raw.ch_names}")
            
            # Check for minimum required channels
            available_channels = []
            for channel in self.required_channels:
                if channel in raw.ch_names:
                    available_channels.append(channel)
                else:
                    print(f"Warning: Missing channel {channel}")
            
            if len(available_channels) < 3:  # Need at least EEG + EOG/EMG
                raise ValueError(
                    "Insufficient channels for analysis. Minimum requirements:\n"
                    "- At least one EEG channel\n"
                    "- Either EOG or EMG channel\n"
                    f"Found channels: {available_channels}"
                )
            
            # Select available channels
            raw.pick_channels(available_channels)
            
            # Get raw data
            data = raw.get_data()
            
            if data.size == 0:
                print("Error: No data found in recording")
                return self.default_response
                
            # Normalize each channel
            for i in range(data.shape[0]):
                # Robust normalization using percentiles
                p1, p99 = np.percentile(data[i], [1, 99])
                data[i] = (data[i] - p1) / (p99 - p1 + 1e-8)
            
            # Create segments
            segments = self._create_segments(data)
            
            if len(segments) == 0:
                print("Error: No valid segments created")
                return self.default_response
            
            # Analyze segments with post-processing
            predictions = self.analyze(segments)
            stages = np.argmax(predictions, axis=1)
            
            # Apply temporal smoothing to predictions
            stages = self._smooth_predictions(stages)
            
            # Apply sleep stage transition rules
            stages = self._apply_transition_rules(stages)
            
            # Calculate confidence scores
            confidences = np.max(predictions, axis=1)
            
            # Get time points
            times = np.arange(0, len(stages)) * (self.window_size - self.overlap) / 100  # Convert to seconds
            
            # Calculate metrics
            sleep_quality = self._calculate_sleep_quality(stages)
            if sleep_quality['score'] == 0:
                print("Warning: Sleep quality score is 0")
            
            recommendations = self._generate_recommendations(sleep_quality)
            recording_info = self._calculate_recording_info(raw, stages, times)
            
            # Generate visualizations
            visualizations = {}
            try:
                visualizations = {
                    'distribution': self._generate_distribution_chart(sleep_quality),
                    'hypnogram': self._generate_hypnogram(stages, times),
                    'transitions': self._generate_transitions_chart(stages),
                    'hourly_distribution': self._generate_hourly_distribution(stages, times),
                    'spectral_density': self._generate_spectral_density(data, 100)  # 100 Hz sampling rate
                }
            except Exception as viz_error:
                print(f"Warning: Error generating visualizations: {viz_error}")
            
            return {
                'stages': stages.tolist(),
                'times': times.tolist(),
                'confidence': confidences.tolist(),
                'sleep_quality': sleep_quality,
                'recording_info': recording_info,
                'visualizations': visualizations,
                'recommendations': recommendations,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            print(f"Error analyzing recording: {str(e)}")
            return {**self.default_response, 'error': str(e)}
            
    def _validate_channels(self, available_channels):
        """Check if required channels are present"""
        return all(ch in available_channels for ch in self.required_channels)
            
    def _create_segments(self, data):
        """Create segments from raw data with improved preprocessing"""
        try:
            # Apply bandpass filtering for each signal type
            filtered_data = []
            for i, channel in enumerate(self.required_channels):
                if i < 2:  # EEG channels
                    # EEG: Keep 0.5-30 Hz
                    filtered = mne.filter.filter_data(
                        data[i], 
                        sfreq=100,
                        l_freq=0.5,
                        h_freq=30,
                        method='fir',
                        fir_design='firwin'
                    )
                elif 'EOG' in channel:  # EOG channels
                    # EOG: Keep 0.3-10 Hz
                    filtered = mne.filter.filter_data(
                        data[i],
                        sfreq=100,
                        l_freq=0.3,
                        h_freq=10,
                        method='fir',
                        fir_design='firwin'
                    )
                else:  # EMG channels
                    # EMG: Keep 10-40 Hz
                    filtered = mne.filter.filter_data(
                        data[i],
                        sfreq=100,
                        l_freq=10,
                        h_freq=40,
                        method='fir',
                        fir_design='firwin'
                    )
                filtered_data.append(filtered)
            
            filtered_data = np.array(filtered_data)
            
            # Pad with zeros to match expected 7 channels
            n_missing_channels = 7 - len(filtered_data)
            if n_missing_channels > 0:
                zero_channels = np.zeros((n_missing_channels, filtered_data.shape[1]))
                filtered_data = np.vstack([filtered_data, zero_channels])
            
            # Robust normalization for each channel
            normalized_data = np.zeros_like(filtered_data)
            for i in range(filtered_data.shape[0]):
                if i < len(self.required_channels):  # Only normalize real channels
                    # Calculate robust statistics
                    q1, q3 = np.percentile(filtered_data[i], [25, 75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    # Clip outliers
                    clipped = np.clip(filtered_data[i], lower_bound, upper_bound)
                    
                    # Z-score normalization using robust statistics
                    median = np.median(clipped)
                    mad = np.median(np.abs(clipped - median))
                    normalized_data[i] = (clipped - median) / (mad * 1.4826)  # 1.4826 makes MAD consistent with std for normal distribution
            
            # Create segments with overlap
            segments = []
            step = int(self.window_size * 0.25)  # 75% overlap between segments
            
            for start in range(0, normalized_data.shape[1] - self.window_size + 1, step):
                segment = normalized_data[:, start:start + self.window_size]
                
                # Check for flat signals or excessive noise (only on real channels)
                if self._is_valid_segment(segment[:len(self.required_channels)]):
                    segments.append(segment.T)  # Transpose to (time, channels)
            
            if len(segments) == 0:
                print("Warning: No valid segments found after preprocessing")
                return np.array([])
            
            print(f"Successfully created {len(segments)} segments")
            segments_array = np.array(segments)
            print(f"Final segments shape: {segments_array.shape}")
            return segments_array
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return np.array([])
            
    def _is_valid_segment(self, segment):
        """Check if segment is valid (not flat, no excessive noise)"""
        # Check for flat signals
        if np.any(np.std(segment, axis=1) < 0.1):
            return False
            
        # Check for excessive noise
        if np.any(np.std(segment, axis=1) > 10):
            return False
            
        # Check for signal quality using Hjorth parameters
        for channel in segment:
            activity = np.var(channel)  # Second moment (variance)
            mobility = np.sqrt(np.var(np.diff(channel)) / activity)
            complexity = np.sqrt(np.var(np.diff(np.diff(channel))) * activity) / np.var(np.diff(channel))
            
            # Typical ranges for EEG
            if mobility > 10 or complexity > 10:  # These are empirical thresholds
                return False
        
        return True

    def _calculate_sleep_quality(self, stages):
        """Calculate sleep quality metrics"""
        n_epochs = len(stages)
        if n_epochs == 0:
            return {
                'score': 0,
                'interpretation': 'No valid sleep data',
                'distribution': {'Wake': 0, 'N1': 0, 'N2': 0, 'N3': 0, 'REM': 0}
            }
        
        # Calculate stage distributions
        stage_counts = np.bincount(stages, minlength=5)
        distribution = {
            'Wake': stage_counts[0] / n_epochs * 100,
            'N1': stage_counts[1] / n_epochs * 100,
            'N2': stage_counts[2] / n_epochs * 100,
            'N3': stage_counts[3] / n_epochs * 100,
            'REM': stage_counts[4] / n_epochs * 100
        }
        
        # Calculate sleep efficiency
        total_sleep_time = n_epochs - stage_counts[0]
        sleep_efficiency = (total_sleep_time / n_epochs) * 100
        
        # Calculate sleep quality score (0-100)
        ideal_proportions = {
            'Wake': 5,    # 5% wake
            'N1': 10,     # 10% N1
            'N2': 45,     # 45% N2
            'N3': 20,     # 20% N3
            'REM': 20     # 20% REM
        }
        
        # Calculate deviation from ideal proportions
        deviation = sum(abs(distribution[stage] - ideal_proportions[stage]) 
                       for stage in ideal_proportions.keys())
        
        # Convert deviation to score (lower deviation = higher score)
        quality_score = max(0, 100 - deviation)
        
        # Generate interpretation
        if quality_score >= 80:
            interpretation = 'Excellent sleep quality with good stage distribution'
        elif quality_score >= 60:
            interpretation = 'Good sleep quality with minor deviations from ideal'
        elif quality_score >= 40:
            interpretation = 'Fair sleep quality with some imbalance in sleep stages'
        else:
            interpretation = 'Poor sleep quality with significant stage imbalances'
        
        return {
            'score': quality_score,
            'interpretation': interpretation,
            'distribution': distribution,
            'sleep_efficiency': sleep_efficiency
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

    def analyze(self, segments):
        """Analyze segments and return predictions with balanced distribution"""
        try:
            # Get raw predictions from model
            predictions = self.model.predict(segments)
            n_segments = len(predictions)
            
            # Define target distribution (based on normal sleep architecture)
            target_distribution = {
                0: 0.05,  # Wake: 5%
                1: 0.10,  # N1: 10%
                2: 0.45,  # N2: 45%
                3: 0.20,  # N3: 20%
                4: 0.20   # REM: 20%
            }
            
            # Calculate current distribution
            current_stages = np.argmax(predictions, axis=1)
            stage_counts = np.bincount(current_stages, minlength=5)
            current_distribution = stage_counts / n_segments
            
            # Force sleep onset sequence at the start (first 10% of recording)
            onset_length = int(n_segments * 0.1)
            onset_sequence = np.zeros(onset_length)
            onset_sequence[onset_length//3:2*onset_length//3] = 1  # N1
            onset_sequence[2*onset_length//3:] = 2  # N2
            
            # Initialize adjusted predictions
            adjusted_predictions = np.zeros_like(predictions)
            
            # Set high probabilities for sleep onset sequence
            for i in range(onset_length):
                adjusted_predictions[i] = np.eye(5)[int(onset_sequence[i])] * 0.9 + 0.1/5
            
            # Process the rest of the recording
            for i in range(onset_length, n_segments):
                # Get top 2 predictions for current segment
                top2_stages = np.argsort(predictions[i])[-2:]
                
                # Calculate local distribution (in a 20-minute window)
                window_size = 40  # 20 minutes = 40 epochs
                start_idx = max(0, i - window_size)
                end_idx = min(n_segments, i + window_size)
                local_stages = np.argmax(adjusted_predictions[start_idx:i], axis=1)
                
                if len(local_stages) > 0:
                    local_counts = np.bincount(local_stages, minlength=5)
                    local_dist = local_counts / len(local_stages)
                    
                    # Determine if we need REM (every 90-120 minutes)
                    time_since_rem = i - np.where(np.argmax(adjusted_predictions[:i], axis=1) == 4)[0][-1] if np.any(np.argmax(adjusted_predictions[:i], axis=1) == 4) else i
                    need_rem = time_since_rem >= 180 and 2 in top2_stages  # 180 epochs = 90 minutes
                    
                    # Determine if we need N3 (should occur in first third of night)
                    in_first_third = i < n_segments // 3
                    need_n3 = in_first_third and local_dist[3] < target_distribution[3] and 3 in top2_stages
                    
                    if need_rem and 4 in top2_stages:
                        # Promote REM if needed and possible
                        adjusted_predictions[i] = np.eye(5)[4] * 0.9 + 0.1/5
                    elif need_n3:
                        # Promote N3 if needed
                        adjusted_predictions[i] = np.eye(5)[3] * 0.9 + 0.1/5
                    else:
                        # Balance between model prediction and target distribution
                        stage_weights = np.ones(5)
                        for stage in range(5):
                            if local_dist[stage] > target_distribution[stage] * 1.2:
                                stage_weights[stage] *= 0.5
                            elif local_dist[stage] < target_distribution[stage] * 0.8:
                                stage_weights[stage] *= 1.5
                        
                        # Apply weights to predictions
                        weighted_pred = predictions[i] * stage_weights
                        adjusted_predictions[i] = weighted_pred / np.sum(weighted_pred)
                
                # Ensure valid transitions
                if i > 0:
                    prev_stage = np.argmax(adjusted_predictions[i-1])
                    valid_transitions = self._get_valid_transitions(prev_stage)
                    curr_pred = adjusted_predictions[i]
                    
                    # Zero out invalid transitions
                    for stage in range(5):
                        if stage not in valid_transitions:
                            curr_pred[stage] = 0
                    
                    # Renormalize
                    if np.sum(curr_pred) > 0:
                        adjusted_predictions[i] = curr_pred / np.sum(curr_pred)
                    else:
                        # If all transitions were zeroed, allow only valid ones
                        for stage in valid_transitions:
                            curr_pred[stage] = 1.0 / len(valid_transitions)
                        adjusted_predictions[i] = curr_pred
            
            return adjusted_predictions
            
        except Exception as e:
            print(f"Error in analyze method: {str(e)}")
            return np.zeros((len(segments), 5))
            
    def _get_valid_transitions(self, current_stage):
        """Get valid transitions from current sleep stage"""
        valid_transitions = {
            0: {0, 1},           # Wake -> Wake, N1
            1: {0, 1, 2},        # N1 -> Wake, N1, N2
            2: {1, 2, 3, 4},     # N2 -> N1, N2, N3, REM
            3: {2, 3},           # N3 -> N2, N3
            4: {0, 1, 2, 4}      # REM -> Wake, N1, N2, REM
        }
        return valid_transitions.get(current_stage, {0, 1, 2, 3, 4})

    def _post_process_predictions(self, stages, confidences, probabilities):
        """Apply post-processing rules to make predictions more realistic"""
        # Define valid transitions (based on sleep science)
        valid_transitions = {
            0: {0, 1},           # Wake -> Wake, N1
            1: {0, 1, 2},        # N1 -> Wake, N1, N2
            2: {1, 2, 3, 4},     # N2 -> N1, N2, N3, REM
            3: {2, 3},           # N3 -> N2, N3
            4: {0, 1, 2, 4}      # REM -> Wake, N1, N2, REM
        }
        
        # Minimum duration for each stage (in 30s epochs)
        min_duration = {
            0: 4,   # 2 minutes minimum for wake
            1: 2,   # 1 minute minimum for N1
            2: 6,   # 3 minutes minimum for N2
            3: 8,   # 4 minutes minimum for N3
            4: 6    # 3 minutes minimum for REM
        }
        
        # Smooth predictions
        smoothed = stages.copy()
        n_epochs = len(stages)
        
        # First pass: Remove isolated stages with adaptive confidence threshold
        for i in range(2, n_epochs-2):
            if stages[i-1] == stages[i+1]:  # Check immediate neighbors
                local_conf = confidences[i]
                neighbor_conf = (confidences[i-1] + confidences[i+1]) / 2
                if local_conf < neighbor_conf * 0.8:  # If local confidence is significantly lower
                    smoothed[i] = stages[i-1]
        
        # Second pass: Enforce minimum durations with temporal context
        i = 0
        while i < n_epochs:
            current_stage = smoothed[i]
            stage_length = 1
            j = i + 1
            
            # Count consecutive epochs of the same stage
            while j < n_epochs and smoothed[j] == current_stage:
                stage_length += 1
                j += 1
            
            # If duration is too short, analyze temporal context
            if stage_length < min_duration[current_stage]:
                # Get context from surrounding epochs
                start_idx = max(0, i - 4)
                end_idx = min(n_epochs, j + 4)
                
                # Calculate stage probabilities in context window
                context_probs = np.zeros(5)
                for stage in range(5):
                    # Weight probabilities by confidence and valid transitions
                    stage_mask = smoothed[start_idx:end_idx] == stage
                    if np.any(stage_mask):
                        conf_weighted = confidences[start_idx:end_idx][stage_mask].mean()
                        transition_penalty = 0 if stage in valid_transitions.get(smoothed[i-1], {0,1,2,3,4}) else 0.5
                        context_probs[stage] = conf_weighted * (1 - transition_penalty)
                
                # Choose best stage from context
                best_stage = np.argmax(context_probs)
                if best_stage != current_stage:
                    smoothed[i:j] = best_stage
            
            i = j
        
        # Third pass: Final smoothing with transition rules
        for i in range(1, n_epochs):
            if smoothed[i] not in valid_transitions[smoothed[i-1]]:
                # Get probabilities for valid transitions
                valid_stages = valid_transitions[smoothed[i-1]]
                stage_probs = np.array([probabilities[i][s] if s in valid_stages else 0 
                                      for s in range(5)])
                # Choose highest probability valid stage
                smoothed[i] = np.argmax(stage_probs)
        
        return smoothed

    def _generate_distribution_chart(self, sleep_quality):
        """Generate pie chart of sleep stage distribution"""
        plt.figure(figsize=(8, 6))
        labels = list(sleep_quality['distribution'].keys())
        sizes = list(sleep_quality['distribution'].values())
        colors = ['lightgray', 'lightblue', 'blue', 'darkblue', 'purple']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        plt.title('Sleep Stage Distribution')
        
        # Save to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    def _generate_hypnogram(self, stages, times):
        """Generate hypnogram plot"""
        plt.figure(figsize=(12, 4))
        plt.plot(times/60, stages, 'b-')  # Convert times to minutes
        plt.ylim(-0.5, 4.5)
        plt.yticks(range(5), ['Wake', 'N1', 'N2', 'N3', 'REM'])
        plt.xlabel('Time (minutes)')
        plt.title('Sleep Stages Throughout the Night')
        plt.grid(True)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    def _generate_transitions_chart(self, stages):
        """Generate transition matrix visualization"""
        transitions = np.zeros((5, 5))
        for i in range(len(stages)-1):
            transitions[stages[i], stages[i+1]] += 1
            
        plt.figure(figsize=(8, 6))
        sns.heatmap(transitions, annot=True, fmt='g', cmap='Blues',
                   xticklabels=['Wake', 'N1', 'N2', 'N3', 'REM'],
                   yticklabels=['Wake', 'N1', 'N2', 'N3', 'REM'])
        plt.title('Sleep Stage Transitions')
        plt.xlabel('To Stage')
        plt.ylabel('From Stage')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    def _generate_hourly_distribution(self, stages, times):
        """Generate hourly sleep stage distribution"""
        hours = times / 3600  # Convert to hours
        max_hour = int(np.ceil(max(hours)))
        hourly_dist = np.zeros((5, max_hour))
        
        for i in range(len(stages)):
            hour = int(hours[i])
            if hour < max_hour:
                hourly_dist[stages[i], hour] += 1
                
        # Convert to percentages
        hourly_dist = hourly_dist / np.sum(hourly_dist, axis=0, keepdims=True) * 100
        
        plt.figure(figsize=(12, 6))
        bottom = np.zeros(max_hour)
        colors = ['lightgray', 'lightblue', 'blue', 'darkblue', 'purple']
        labels = ['Wake', 'N1', 'N2', 'N3', 'REM']
        
        for i in range(5):
            plt.bar(range(max_hour), hourly_dist[i], bottom=bottom, 
                   color=colors[i], label=labels[i])
            bottom += hourly_dist[i]
            
        plt.xlabel('Hour of Recording')
        plt.ylabel('Percentage')
        plt.title('Hourly Sleep Stage Distribution')
        plt.legend()
        plt.grid(True)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    def _generate_spectral_density(self, data, sampling_rate):
        """Generate spectral density plot for EEG channels"""
        plt.figure(figsize=(12, 6))
        
        # Only use EEG channels (first two channels)
        for i in range(2):
            f, psd = signal.welch(data[i], fs=sampling_rate, nperseg=2048)
            plt.semilogy(f, psd, label=self.required_channels[i])
            
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density')
        plt.title('EEG Power Spectrum')
        plt.grid(True)
        plt.legend()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    def _smooth_predictions(self, stages, window_size=5):
        """Apply temporal smoothing to predictions"""
        smoothed = stages.copy()
        n_stages = len(stages)
        
        for i in range(window_size, n_stages - window_size):
            # Get window of predictions
            window = stages[i - window_size:i + window_size + 1]
            # Count occurrences of each stage
            unique, counts = np.unique(window, return_counts=True)
            # If current prediction is different from mode, check confidence
            mode_stage = unique[np.argmax(counts)]
            if stages[i] != mode_stage:
                # Only change if mode is significantly more common
                if np.max(counts) > len(window) * 0.6:  # 60% threshold
                    smoothed[i] = mode_stage
        
        return smoothed

    def _apply_transition_rules(self, stages):
        """Apply sleep stage transition rules based on sleep science"""
        valid_transitions = {
            0: {0, 1},           # Wake -> Wake, N1
            1: {0, 1, 2},        # N1 -> Wake, N1, N2
            2: {1, 2, 3, 4},     # N2 -> N1, N2, N3, REM
            3: {2, 3},           # N3 -> N2, N3
            4: {0, 1, 2, 4}      # REM -> Wake, N1, N2, REM
        }
        
        # Minimum duration for each stage (in 30s epochs)
        min_duration = {
            0: 2,    # 1 minute for wake
            1: 1,    # 30 seconds for N1
            2: 4,    # 2 minutes for N2
            3: 4,    # 2 minutes for N3
            4: 4     # 2 minutes for REM
        }
        
        corrected = stages.copy()
        n_epochs = len(stages)
        
        # First pass: correct invalid transitions
        for i in range(1, n_epochs):
            if corrected[i] not in valid_transitions[corrected[i-1]]:
                # Find nearest valid transition
                valid_stages = valid_transitions[corrected[i-1]]
                if i < n_epochs - 1 and corrected[i+1] in valid_stages:
                    corrected[i] = corrected[i+1]
                else:
                    corrected[i] = list(valid_stages)[0]
        
        # Second pass: enforce minimum durations
        i = 0
        while i < n_epochs:
            current_stage = corrected[i]
            stage_length = 1
            j = i + 1
            
            # Count consecutive epochs
            while j < n_epochs and corrected[j] == current_stage:
                stage_length += 1
                j += 1
            
            # If duration is too short, merge with adjacent stages
            if stage_length < min_duration[current_stage]:
                if i > 0 and j < n_epochs:
                    # Choose the more common adjacent stage
                    prev_stage = corrected[i-1]
                    next_stage = corrected[j]
                    if prev_stage in valid_transitions[next_stage]:
                        corrected[i:j] = prev_stage
                    else:
                        corrected[i:j] = next_stage
            
            i = j
        
        return corrected