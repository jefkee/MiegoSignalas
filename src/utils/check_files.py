import os
import mne

def validate_edf_file(file_path):
    """Validate EDF file format and content"""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return False, "File does not exist"
            
        # Check file extension
        if not file_path.endswith('.edf'):
            return False, "Invalid file format (must be .edf)"
            
        # Try to load file with MNE
        raw = mne.io.read_raw_edf(file_path, preload=True)
        
        # Check if file has data
        if raw.n_times == 0:
            return False, "File contains no data"
            
        return True, "File is valid"
        
    except Exception as e:
        return False, f"Error validating file: {str(e)}"
        
def check_required_channels(raw_data, required_channels):
    """Check if all required channels are present"""
    missing_channels = []
    
    for channel in required_channels:
        if channel not in raw_data.ch_names:
            missing_channels.append(channel)
            
    if missing_channels:
        return False, f"Missing required channels: {', '.join(missing_channels)}"
    
    return True, "All required channels present" 