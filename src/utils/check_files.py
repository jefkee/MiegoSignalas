import os

def check_data_directory(data_dir):
    """Check and print files in data directory"""
    print(f"\nChecking directory: {data_dir}")
    print("-" * 50)
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist!")
        return
    
    files = os.listdir(data_dir)
    print(f"Total files found: {len(files)}")
    
    # Check all possible file patterns
    psg_files = [f for f in files if 'PSG' in f]
    hypno_patterns = ['Hypnogram', 'hypnogram', 'HYP', 'hyp']
    
    print(f"\nPSG files ({len(psg_files)}):")
    for f in psg_files:
        print(f"  - {f}")
        base = f.replace('-PSG.edf', '')
        
        # Check for corresponding hypnogram with any pattern
        hypno_found = False
        for pattern in hypno_patterns:
            possible_names = [
                f"{base}-{pattern}.edf",
                f"{base.replace('0', 'J')}-{pattern}.edf",
                f"{base}{pattern}.edf",
                f"{base}.{pattern}"
            ]
            for hypno_name in possible_names:
                if hypno_name in files:
                    print(f"    Found hypnogram: {hypno_name}")
                    hypno_found = True
                    break
        if not hypno_found:
            print(f"    No hypnogram found for {base}")

if __name__ == "__main__":
    data_dir = os.path.join('data', 'raw')
    check_data_directory(data_dir) 