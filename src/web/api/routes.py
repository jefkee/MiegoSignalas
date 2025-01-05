from flask import Blueprint, request, jsonify
from src.models import SleepAnalyzer
from src.utils.file_handler import FileHandler
from werkzeug.utils import secure_filename
import os

api = Blueprint('api', __name__)
file_handler = FileHandler()
analyzer = SleepAnalyzer()

@api.route('/analyze', methods=['POST'])
def analyze_recording():
    try:
        # Check if the request has files
        if not request.files:
            return jsonify({'error': 'No files were uploaded'}), 400
        
        # Get PSG file from request
        psg_file = request.files.get('psg_file')
        if not psg_file or not psg_file.filename:
            return jsonify({'error': 'PSG file is required'}), 400
        
        if not psg_file.filename.endswith('-PSG.edf'):
            return jsonify({'error': 'File must end with -PSG.edf'}), 400
        
        # Ensure uploads directory exists
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save PSG file
        psg_filename = secure_filename(psg_file.filename)
        psg_path = os.path.join(upload_dir, psg_filename)
        psg_file.save(psg_path)
        
        # Analyze recording
        results = analyzer.analyze_recording(psg_path)
        
        # Clean up files
        os.remove(psg_path)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500