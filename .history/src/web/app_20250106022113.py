from flask import Flask, request, jsonify
from src.utils.visualization import SleepVisualizer
from flask_cors import CORS
from src.web.api.routes import api

def create_app():
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    
    # Configure CORS
    CORS(app, resources={
        r"/api/*": {
            "origins": "*",
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # Register blueprints
    app.register_blueprint(api, url_prefix='/api')
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    # Create required directories
    import os
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    @app.route('/api/analyze', methods=['POST'])
    def analyze():
        try:
            # Failo gavimas ir apdorojimas
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            
            file = request.files['file']
            
            # Analizės vykdymas
            # ...
            
            # Vizualizacijų kūrimas
            visualizer = SleepVisualizer()
            visualizations = visualizer.create_visualizations(predictions, raw_data)
            
            # Rezultatų grąžinimas
            response = {
                'sleep_stages': {
                    'images': visualizations,  # Čia turėtų būti dictionary su 'hypnogram', 'distribution', 'transitions' raktais
                    'predictions': predictions.tolist()
                },
                'sleep_quality': {
                    'score': quality_score,
                    'interpretation': quality_interpretation
                },
                'recommendations': recommendations
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000) 