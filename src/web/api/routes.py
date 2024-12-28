from flask import Blueprint, request, jsonify, send_from_directory
from src.models import SleepAnalyzer
from src.utils.file_handler import FileHandler
from src.utils.visualization import SleepVisualizer
from src.utils.report_generator import ReportGenerator
from werkzeug.utils import secure_filename
import os
from flask_cors import CORS

api = Blueprint('api', __name__)
CORS(api)
file_handler = FileHandler()
analyzer = SleepAnalyzer()
visualizer = SleepVisualizer()
report_gen = ReportGenerator()

def get_quality_interpretation(score):
    """Interpret sleep quality score"""
    if score >= 90:
        return "Excellent sleep quality"
    elif score >= 75:
        return "Good sleep quality"
    elif score >= 60:
        return "Fair sleep quality"
    else:
        return "Poor sleep quality"

def generate_hypnogram(stages):
    """Generate hypnogram visualization"""
    return visualizer.create_hypnogram(stages)

def generate_recommendations(analysis):
    """Generate sleep recommendations"""
    return report_gen.generate_recommendations(analysis)

@api.route('/analyze', methods=['POST'])
def analyze_recording():
    try:
        # Debug logging
        print("\n=== Starting Analysis ===")
        print("Files in request:", request.files)
        print("Form data:", request.form)
        print("Request headers:", dict(request.headers))
        
        # Check if the request has the correct content type
        if not request.content_type or 'multipart/form-data' not in request.content_type:
            print(f"Invalid content type: {request.content_type}")
            return jsonify({'error': 'Request must be multipart/form-data'}), 400
        
        # Check if any files were sent
        if not request.files:
            print("No files in request")
            return jsonify({'error': 'No files were uploaded'}), 400
        
        # Get PSG file from request
        psg_file = request.files.get('psg_file')
        print(f"PSG file object: {psg_file}")
        print(f"PSG filename: {psg_file.filename if psg_file else 'None'}")
        
        # Validate PSG file
        if not psg_file or not psg_file.filename:
            print("PSG file missing or invalid")
            return jsonify({'error': 'PSG file is required'}), 400
        
        if not psg_file.filename.endswith('-PSG.edf'):
            print(f"Invalid file format: {psg_file.filename}")
            return jsonify({'error': 'File must end with -PSG.edf'}), 400
        
        # Ensure uploads directory exists
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        
        try:
            # Save PSG file with secure filename
            psg_filename = secure_filename(psg_file.filename)
            psg_path = os.path.join(upload_dir, psg_filename)
            print(f"Saving file to: {psg_path}")
            psg_file.save(psg_path)
            
            if not os.path.exists(psg_path):
                print("File was not saved successfully")
                return jsonify({'error': 'Failed to save uploaded file'}), 500
                
            print(f"File saved successfully. Size: {os.path.getsize(psg_path)} bytes")
            
            # Initialize analyzer
            print("\nInitializing analyzer...")
            analyzer = SleepAnalyzer()
            
            # Analyze recording
            print("\nAnalyzing recording...")
            raw_results = analyzer.analyze_recording(psg_path)
            
            if not raw_results:
                print("Error: Analysis produced no results")
                return jsonify({'error': 'Analysis failed to produce results'}), 500
            
            print("\nCreating visualizations...")
            # Get data for visualizations
            raw_data = raw_results.get('raw_data', None)
            stages = raw_results.get('stages', [])
            
            # Create visualizations
            visualizations = visualizer.create_visualizations(stages, raw_data)
            
            # Format results
            print("\nFormatting results...")
            results = {
                'sleep_quality': {
                    'score': raw_results.get('quality_score', 0),
                    'interpretation': get_quality_interpretation(raw_results.get('quality_score', 0))
                },
                'sleep_stages': visualizations,
                'timing': raw_results.get('timing', {}),
                'recommendations': report_gen.generate_recommendations(raw_results)
            }
            
            print("\nSending response...")
            return jsonify(results)
            
        except Exception as e:
            print(f"\nError during analysis: {str(e)}")
            return jsonify({'error': str(e)}), 500
            
        finally:
            # Clean up files
            print("\nCleaning up files...")
            try:
                if psg_path and os.path.exists(psg_path):
                    os.remove(psg_path)
                    print(f"Removed PSG file: {psg_path}")
            except Exception as e:
                print(f"Error removing PSG file: {e}")
                
    except Exception as e:
        print(f"\nError in route handler: {str(e)}")
        return jsonify({'error': f"Upload error: {str(e)}"}), 500

@api.route('/visualizations/<path:filename>')
def serve_visualization(filename):
    """Serve visualization files"""
    vis_dir = os.path.join('models', 'visualizations')
    
    # Check which subdirectory the file belongs to
    for subdir in ['curves', 'confusion', 'roc', 'models']:
        file_path = os.path.join(vis_dir, subdir, filename)
        if os.path.exists(file_path):
            return send_from_directory(os.path.join(vis_dir, subdir), filename)
    
    # Check root visualization directory
    if os.path.exists(os.path.join(vis_dir, filename)):
        return send_from_directory(vis_dir, filename)
    
    return jsonify({'error': 'File not found'}), 404