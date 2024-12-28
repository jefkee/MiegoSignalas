import mne
import numpy as np
from .classifier import SleepStageClassifier
import os

class SleepAnalyzer:
    """Neural network-based sleep stage analyzer"""
    
    def __init__(self):
        # Initialize the analyzer with a neural network model
        self.model = SleepStageClassifier()  # Create classifier instance
        self.model.load_model('models/sleep_classifier.h5')  # Load trained weights
        
        # Define required EEG channels for analysis
        self.required_channels = [
            'EEG Fpz-Cz',      # Frontal EEG channel
            'EEG Pz-Oz',       # Occipital EEG channel
            'EOG horizontal',   # Eye movement channel
            'Resp oro-nasal',   # Breathing channel
            'EMG submental',    # Chin muscle activity
            'Temp rectal',      # Body temperature
            'Event marker'      # Event annotations
        ]
        
    def analyze_recording(self, psg_path, hypno_path=None):
        try:
            # Load EDF file
            raw_data = mne.io.read_raw_edf(psg_path, preload=True)
            print(f"Loading EDF file: {psg_path}")
            print(f"Available channels: {raw_data.ch_names}")
            
            # Load hypnogram file if provided
            hypno_data = None
            if hypno_path:
                try:
                    hypno_data = mne.read_annotations(hypno_path)
                except Exception as e:
                    print(f"Warning: Could not load hypnogram: {e}")
            
            # Validate channels
            if not self._validate_channels(raw_data.ch_names):
                missing_channels = set(self.required_channels) - set(raw_data.ch_names)
                raise ValueError(
                    "Missing required channels in the recording:\n"
                    f"Missing: {list(missing_channels)}\n"
                    f"Found: {raw_data.ch_names}\n"
                    "Please ensure you're using a compatible PSG recording."
                )
            
            # Select required channels in correct order
            raw_data.pick_channels(self.required_channels)
            
            # Get raw data
            data = raw_data.get_data()
            
            if data.size == 0:
                raise ValueError("No data found in the recording")
                
            if data.shape[0] != len(self.required_channels):
                raise ValueError(f"Expected {len(self.required_channels)} channels, got {data.shape[0]}")
            
            # Normalize each channel
            for i in range(data.shape[0]):
                data[i] = (data[i] - np.mean(data[i])) / (np.std(data[i]) + 1e-8)
            
            # Create segments
            segments = self._create_segments(data)
            
            if len(segments) == 0:
                raise ValueError("Recording too short for analysis")
                
            print(f"Input features shape: {segments.shape}")
            
            # Predict sleep stages
            predictions = self.model.predict(segments)
            stages = np.argmax(predictions, axis=1)
            
            # Calculate quality metrics
            quality_score = self._calculate_quality_score(stages)
            recommendations = self._get_recommendations(quality_score, stages)
            
            print("\nStarting timing calculations...")
            # Calculate timing information
            epoch_duration = 30  # seconds
            total_duration = len(stages) * epoch_duration
            hours = total_duration // 3600
            minutes = (total_duration % 3600) // 60
            print(f"Total duration calculated: {hours:02d}:{minutes:02d}")

            # Get recording start time
            try:
                print("\nGetting recording start time...")
                recording_start = raw_data.info.get('meas_date', None)
                print(f"Raw meas_date: {recording_start}")
                if recording_start is not None:
                    recording_start_str = recording_start.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    print("No meas_date, checking annotations...")
                    annot = raw_data.annotations
                    print(f"Annotations found: {len(annot.onset)}")
                    if len(annot.onset) > 0:
                        start_time = raw_data.annotations.onset[0]
                        recording_start_str = f"Recording started at {int(start_time // 3600):02d}:{int((start_time % 3600) // 60):02d}"
                    else:
                        recording_start_str = "Not available"
            except Exception as e:
                print(f"Error getting recording start time: {e}")
                recording_start_str = "Not available"

            print("\nCalculating sleep statistics...")
            # Calculate sleep stages statistics
            stage_counts = np.bincount(stages, minlength=5)
            print(f"Stage counts: {stage_counts}")
            wake_epochs = stage_counts[0]
            sleep_epochs = sum(stage_counts[1:])  # All non-wake epochs
            print(f"Wake epochs: {wake_epochs}, Sleep epochs: {sleep_epochs}")

            # Calculate sleep onset latency (time to first sleep epoch)
            sleep_onset_idx = next((i for i, stage in enumerate(stages) if stage != 0), None)
            if sleep_onset_idx is not None:
                sleep_latency = sleep_onset_idx * epoch_duration
                sleep_onset_str = f"{sleep_latency // 60:02d}:{sleep_latency % 60:02d}"
            else:
                sleep_onset_str = "No sleep detected"

            # Find last sleep epoch
            last_sleep_idx = len(stages) - next((i for i, stage in enumerate(reversed(stages)) if stage != 0), len(stages))
            
            # Calculate actual sleep time (from first to last sleep epoch)
            if sleep_onset_idx is not None and last_sleep_idx > sleep_onset_idx:
                sleep_duration = (last_sleep_idx - sleep_onset_idx) * epoch_duration
                actual_sleep_time = sum(1 for s in stages[sleep_onset_idx:last_sleep_idx] if s != 0) * epoch_duration
                sleep_efficiency = (actual_sleep_time / sleep_duration * 100)
            else:
                sleep_duration = 0
                actual_sleep_time = 0
                sleep_efficiency = 0

            # Format times
            total_duration = len(stages) * epoch_duration
            timing_info = {
                'recording_start': raw_data.info.get('meas_date', 'Not available').strftime('%Y-%m-%d %H:%M:%S'),
                'total_duration': f"{total_duration // 3600:02d}:{(total_duration % 3600) // 60:02d}",
                'total_sleep_time': f"{actual_sleep_time // 3600:02d}:{(actual_sleep_time % 3600) // 60:02d}",
                'sleep_onset': f"After {sleep_onset_str}" if sleep_onset_idx is not None else "No sleep detected",
                'wake_time': f"After {(last_sleep_idx * epoch_duration) // 3600:02d}:{((last_sleep_idx * epoch_duration) % 3600) // 60:02d}" if last_sleep_idx > 0 else "No wake detected",
                'sleep_efficiency': f"{sleep_efficiency:.1f}%"
            }

            print("\nFinal timing info:", timing_info)

            return {
                'quality_score': quality_score,
                'stages': stages.tolist(),
                'stage_probabilities': predictions.tolist(),
                'raw_data': data,
                'timing': timing_info,
                'recommendations': recommendations
            }
            
        except Exception as e:
            print(f"Error analyzing recording: {str(e)}")
            raise
            
    def _validate_channels(self, available_channels):
        """Check if required channels are present"""
        return all(ch in available_channels for ch in self.required_channels)
            
    def _create_segments(self, data):
        """Create 30-second segments from raw data"""
        segment_size = 3000  # 30 seconds at 100Hz
        n_segments = data.shape[1] // segment_size
        segments = []
        
        for i in range(n_segments):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size
            segment = data[:, start_idx:end_idx].T
            segments.append(segment)
            
        return np.array(segments)
        
    def _calculate_quality_score(self, stages):
        """Calculate sleep quality score based on stage distribution"""
        total_epochs = len(stages)
        if total_epochs == 0:
            return 0
            
        # Count stages
        stage_counts = np.bincount(stages, minlength=5)
        
        # Calculate percentages
        wake_percent = stage_counts[0] / total_epochs * 100
        rem_percent = stage_counts[4] / total_epochs * 100
        deep_sleep_percent = (stage_counts[3]) / total_epochs * 100
        
        # Score components
        efficiency_score = max(0, 100 - wake_percent)  # Less wake time is better
        rem_score = max(0, 100 - abs(rem_percent - 25))  # Ideal REM is ~25%
        deep_sleep_score = max(0, 100 - abs(deep_sleep_percent - 20))  # Ideal deep sleep is ~20%
        
        # Weighted average
        quality_score = (
            0.4 * efficiency_score +
            0.3 * rem_score +
            0.3 * deep_sleep_score
        )
        
        return round(quality_score, 1)

    def _get_recommendations(self, quality_score, stages):
        """Generate specific recommendations based on sleep quality score and stages"""
        recommendations = []
        
        # Calculate percentages
        total_epochs = len(stages)
        stage_percentages = np.bincount(stages, minlength=5) / total_epochs * 100
        
        if quality_score >= 90:
            recommendations.extend([
                "Excellent sleep quality! Keep maintaining your current sleep habits.",
                "Your sleep architecture shows optimal patterns.",
                "Continue with your current bedtime routine."
            ])
            
        elif quality_score >= 75:
            recommendations.extend([
                "Good sleep quality. Minor improvements possible:",
                "Try to maintain more consistent sleep and wake times",
                "Consider reducing screen time before bed"
            ])
            
            if stage_percentages[0] > 10:  # Too much wake time
                recommendations.append("Try to reduce caffeine intake after 2 PM")
                
        elif quality_score >= 60:
            recommendations.extend([
                "Fair sleep quality. Areas for improvement:",
                "Establish a regular bedtime routine",
                "Ensure your bedroom is dark and cool",
                "Consider relaxation techniques before bed"
            ])
            
            if stage_percentages[3] < 15:  # Low deep sleep
                recommendations.extend([
                    "Exercise earlier in the day to promote deep sleep",
                    "Avoid heavy meals close to bedtime"
                ])
                
            if stage_percentages[4] < 20:  # Low REM sleep
                recommendations.extend([
                    "Try to reduce stress before bedtime",
                    "Maintain a consistent wake-up time"
                ])
                
        else:  # score < 60
            recommendations.extend([
                "Poor sleep quality. Priority improvements needed:",
                "Consult a sleep specialist for professional evaluation",
                "Keep a sleep diary to track patterns",
                "Establish strict sleep schedule"
            ])
            
            if stage_percentages[0] > 20:  # Very high wake time
                recommendations.extend([
                    "Avoid naps during the day",
                    "Create a more sleep-friendly environment",
                    "Consider sleep restriction therapy"
                ])
                
            if stage_percentages[3] < 10:  # Very low deep sleep
                recommendations.extend([
                    "Increase physical activity during the day",
                    "Avoid alcohol before bedtime",
                    "Consider cognitive behavioral therapy for insomnia"
                ])
        
        return recommendations