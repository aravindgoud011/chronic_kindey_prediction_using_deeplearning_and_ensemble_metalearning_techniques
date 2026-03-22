from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class Doctor(UserMixin, db.Model):
    """Doctor user model for authentication"""
    __tablename__ = 'doctors'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(200), nullable=False)
    full_name = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to predictions
    predictions = db.relationship('PredictionResult', backref='doctor', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if provided password matches hash"""
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<Doctor {self.username}>'


class PredictionResult(db.Model):
    """Model to store CKD prediction results"""
    __tablename__ = 'prediction_results'
    
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id'), nullable=False, index=True)
    patient_name = db.Column(db.String(120), nullable=False)
    
    # Clinical inputs
    age = db.Column(db.Float, nullable=False)
    bp = db.Column(db.Float, nullable=False)
    sg = db.Column(db.Float, nullable=False)
    al = db.Column(db.Float, nullable=False)
    su = db.Column(db.Float, nullable=False)
    bgr = db.Column(db.Float, nullable=False)
    bu = db.Column(db.Float, nullable=False)
    sc = db.Column(db.Float, nullable=False)
    sod = db.Column(db.Float, nullable=False)
    pot = db.Column(db.Float, nullable=False)
    hemo = db.Column(db.Float, nullable=False)
    pcv = db.Column(db.Float, nullable=False)
    wbcc = db.Column(db.Float, nullable=False)
    rbcc = db.Column(db.Float, nullable=False)
    hypertension = db.Column(db.Integer, nullable=False)
    
    # Prediction result
    prediction = db.Column(db.String(255), nullable=False)  # "CKD Detected" or "No CKD"
    confidence = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f'<PredictionResult {self.patient_name} - {self.created_at}>'
