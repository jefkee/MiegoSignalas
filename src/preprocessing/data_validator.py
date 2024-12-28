class DataValidator:
    @staticmethod
    def validate_edf(raw_data):
        """Validate EDF file data"""
        validation = {
            'status': True,
            'messages': []
        }
        
        # Check sampling rate
        if raw_data.info['sfreq'] != 100:
            validation['messages'].append(f"Warning: Sampling rate is {raw_data.info['sfreq']}Hz, expected 100Hz")
        
        # Check channel presence
        required_channels = ['EEG Fpz-Cz', 'EEG Pz-Oz']
        missing_channels = [ch for ch in required_channels if ch not in raw_data.ch_names]
        if missing_channels:
            validation['status'] = False
            validation['messages'].append(f"Error: Missing required channels: {missing_channels}")
        
        # Check recording length
        min_duration = 60 * 60  # 1 hour in seconds
        if raw_data.times[-1] < min_duration:
            validation['status'] = False
            validation['messages'].append(f"Error: Recording too short ({raw_data.times[-1]/60:.1f} minutes)")
        
        return validation 