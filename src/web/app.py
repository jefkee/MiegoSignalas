from flask import Flask, render_template, send_from_directory
from flask_cors import CORS
from src.web.api.routes import api
import os

app = Flask(__name__, 
    static_folder='static',
    template_folder='templates'
)
CORS(app)

# Register blueprints
app.register_blueprint(api, url_prefix='/api')

# Root route
@app.route('/')
def index():
    return render_template('index.html')

# Favicon route
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.svg',
        mimetype='image/svg+xml'
    )

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 