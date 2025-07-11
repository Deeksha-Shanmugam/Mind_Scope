# app/models/schema.py
from datetime import datetime
from app.models import db  # ✅ Import shared instance

class PredictionBatch(db.Model):
    __tablename__ = 'prediction_batches'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    total_predictions = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='batch', lazy=True, cascade='all, delete-orphan')

class Prediction(db.Model):
    __tablename__ = 'predictions'
    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.Integer, db.ForeignKey('prediction_batches.id'), nullable=False)
    original_text = db.Column(db.Text, nullable=False)
    predicted_label = db.Column(db.String(100), nullable=False)
    confidence_score = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class SinglePrediction(db.Model):
    __tablename__ = 'single_predictions'
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    predicted_label = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
