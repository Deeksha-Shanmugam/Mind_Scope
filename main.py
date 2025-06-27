from flask import Flask
import os
from app.routes.dashboard import dashboard_bp


def create_app():
    app = Flask(__name__, template_folder=os.path.join('app', 'templates'))

    app.secret_key = os.environ.get('SECRET_KEY', 'fallback-secret-key-for-development')
    # Register Blueprints
    app.register_blueprint(dashboard_bp)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
