import os
import uuid
from werkzeug.utils import secure_filename
import time

class FileHandler:
    def __init__(self, upload_dir='uploads'):
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)
    
    def save_uploaded_file(self, file):
        """Save uploaded file and return path"""
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(self.upload_dir, unique_filename)
        
        file.save(file_path)
        return file_path
    
    def cleanup_old_files(self, max_age_hours=24):
        """Remove old uploaded files"""
        current_time = time.time()
        for filename in os.listdir(self.upload_dir):
            file_path = os.path.join(self.upload_dir, filename)
            if os.path.getmtime(file_path) < current_time - (max_age_hours * 3600):
                os.remove(file_path) 