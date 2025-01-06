from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import numpy as np
from werkzeug.utils import secure_filename
from src.models import SleepAnalyzer
from src.utils.visualization import SleepVisualizer

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'edf'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
            
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        analyzer = SleepAnalyzer()
        visualizer = SleepVisualizer()
        
        try:
            # Load and analyze the data
            raw_data = analyzer.load_data(filepath)
            predictions = analyzer.analyze(raw_data)
            
            # Generate visualizations
            viz_data = visualizer.create_visualizations(predictions, raw_data)
            timing_info = analyzer.calculate_timing(predictions)
            
            # Evaluate sleep quality
            quality_score, quality_interpretation = analyzer.evaluate_quality(predictions)
            
            # Generate recommendations
            recommendations = analyzer.generate_recommendations(predictions, timing_info)
            
            results = {
                'sleep_stages': {
                    'stages': predictions.tolist(),
                    'distribution_values': viz_data['distribution_values'],
                    'time_points': viz_data['time_points'],
                    'transitions': viz_data['transitions']
                },
                'sleep_quality': {
                    'score': quality_score,
                    'interpretation': quality_interpretation
                },
                'timing': timing_info,
                'recommendations': recommendations
            }
            
            # Clean up the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
                
            return jsonify(results)
            
        except Exception as e:
            # Clean up the uploaded file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            raise e
            
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True) 