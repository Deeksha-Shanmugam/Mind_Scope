# app/__init__.py
from flask import Flask
from app.models import db  # ✅ shared instance
import os

def create_app():
    app = Flask(__name__, template_folder=os.path.join( 'templates'))

    app.secret_key = os.environ.get('SECRET_KEY', 'your-development-secret-key-change-in-production')

    basedir = os.path.abspath(os.path.dirname(__file__))
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL') or \
        f'sqlite:///{os.path.join(basedir, "../predictions.db")}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)  # ✅ Register app with db

    from app.routes.dashboard import dashboard_bp
    app.register_blueprint(dashboard_bp)

    with app.app_context():
        db.create_all()

    return app
