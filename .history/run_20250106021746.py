import os
from flask import Flask, send_from_directory
from flask_cors import CORS
from src.web.backend.app import app

# Nustatome kelią iki frontend build katalogo
frontend_build_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', 'web', 'frontend', 'build'))

# Pridedame maršrutą statiniams failams
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(frontend_build_path, path)):
        return send_from_directory(frontend_build_path, path)
    else:
        return send_from_directory(frontend_build_path, 'index.html')

if __name__ == '__main__':
    # Patikriname ar egzistuoja frontend build katalogas
    if not os.path.exists(frontend_build_path):
        print("WARNING: Frontend build directory not found. Please run 'npm run build' in the frontend directory first.")
        print("Building frontend...")
        os.system(f"cd {os.path.join(os.path.dirname(__file__), 'src', 'web', 'frontend')} && npm install && npm run build")
    
    # Paleidžiame serverį
    app.run(debug=True) 