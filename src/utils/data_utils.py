import os
import mne
from typing import List, Tuple

def find_edf_pairs(data_dir: str) -> Tuple[List[str], List[str]]:
    """
    Find matching PSG and hypnogram files
    """
    all_files = os.listdir(data_dir)
    psg_files = sorted([f for f in all_files if 'PSG' in f])
    hypno_files = sorted([f for f in all_files if 'Hypnogram' in f])
    
    return psg_files, hypno_files

def load_edf_file(file_path: str, preload: bool = True) -> mne.io.Raw:
    """
    Load EDF file safely with error handling
    """
    try:
        raw = mne.io.read_raw_edf(file_path, preload=preload)
        return raw
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
