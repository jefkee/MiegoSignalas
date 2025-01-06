from flask import Flask, request, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        # ... existing code ...

        # Formatuojame rezultatus
        results = {
            'sleep_stages': visualizer.create_visualizations(predictions, raw_data),
            'sleep_quality': {
                'score': quality_score,
                'interpretation': quality_interpretation
            },
            'timing': timing_info,
            'recommendations': recommendations
        }

        # Konvertuojame į JSON ir grąžiname
        return jsonify(results)

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500 