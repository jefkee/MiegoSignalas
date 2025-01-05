from src.web.app import app

if __name__ == '__main__':
    # Ensure required directories exist
    import os
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Run app
    app.run(debug=True, port=5000) 