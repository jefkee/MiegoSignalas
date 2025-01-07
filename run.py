import os
from src.web.app import create_app

if __name__ == '__main__':
    # Ensure all required directories exist
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs(os.path.join('models', 'visualizations'), exist_ok=True)
    
    # Create and run app
    app = create_app()
    # app.run(debug=True, host='0.0.0.0', port=5000) 
    app.run(debug=False, host='0.0.0.0', port=5000) 