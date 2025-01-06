from flask import Flask, render_template
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
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000) 