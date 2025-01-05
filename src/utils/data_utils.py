import os

def find_matching_files(data_dir):
    """Find matching PSG and hypnogram files in the data directory"""
    matching_files = []
    
    # Get all PSG files
    psg_files = [f for f in os.listdir(data_dir) if f.endswith('E0-PSG.edf')]
    
    for psg_file in psg_files:
        # Get corresponding hypnogram file
        base_name = psg_file.replace('E0-PSG.edf', '')
        hypno_file = f"{base_name}EC-Hypnogram.edf"
        
        # Check if hypnogram exists
        if hypno_file in os.listdir(data_dir):
            matching_files.append((
                os.path.join(data_dir, psg_file),
                os.path.join(data_dir, hypno_file)
            ))
            
    return matching_files
