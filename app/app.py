from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from app.models import db, Doctor, PredictionResult
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
import re

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{BASE_DIR}/ckd_predictor.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'ckd-predictor-secret-key-2024-change-in-production'
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['PERMANENT_SESSION_LIFETIME'] = 86400 * 7  # 7 days

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load ML models
SCALER_PATH = os.path.join(BASE_DIR, "../artifacts/scalers/scaler.pkl")
META_SCALER_PATH = os.path.join(BASE_DIR, "../artifacts/scalers/meta_scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "../artifacts/models/encoder_model.keras")
TABTRANSFORMER_PATH = os.path.join(BASE_DIR, "../artifacts/models/tabtransformer_model.keras")
META_MODEL_PATH = os.path.join(BASE_DIR, "../artifacts/models/meta_model_lr.pkl")

scaler = joblib.load(SCALER_PATH)
meta_scaler = joblib.load(META_SCALER_PATH)
encoder = load_model(ENCODER_PATH)
tabtransformer = load_model(TABTRANSFORMER_PATH)
meta_model = joblib.load(META_MODEL_PATH)

THRESHOLD = 0.55   # stable ~85%

# Create tables
with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return Doctor.query.get(int(user_id))

def is_valid_username(username):
    """Validate username format"""
    return bool(re.match(r'^[a-zA-Z0-9_]{3,20}$', username))

def is_valid_email(email):
    """Validate email format"""
    return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email))

# ==================== AUTHENTICATION ROUTES ====================

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Doctor registration"""
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        # Validation
        if not all([full_name, username, email, password]):
            return render_template('register.html', error='All fields are required')

        if not is_valid_username(username):
            return render_template('register.html', error='Username must be 3-20 characters (letters, numbers, underscore)')

        if not is_valid_email(email):
            return render_template('register.html', error='Invalid email format')

        if len(password) < 6:
            return render_template('register.html', error='Password must be at least 6 characters')

        if password != confirm_password:
            return render_template('register.html', error='Passwords do not match')

        # Check if user exists
        if Doctor.query.filter_by(username=username).first():
            return render_template('register.html', error='Username already taken')

        if Doctor.query.filter_by(email=email).first():
            return render_template('register.html', error='Email already registered')

        # Create new doctor
        doctor = Doctor(email=email, username=username, full_name=full_name)
        doctor.set_password(password)
        
        db.session.add(doctor)
        db.session.commit()

        flash(f'Registration successful! Welcome, {full_name}. Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Doctor login"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username_or_email = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username_or_email or not password:
            return render_template('login.html', error='Username/Email and password are required')

        # Find doctor by username or email
        doctor = Doctor.query.filter(
            (Doctor.username == username_or_email) | (Doctor.email == username_or_email)
        ).first()

        if doctor and doctor.check_password(password):
            login_user(doctor, remember=True)
            flash(f'Welcome back, {doctor.full_name}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error='Invalid username/email or password')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """Doctor logout"""
    logout_user()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('login'))

# ==================== PREDICTION ROUTES ====================

@app.route('/')
def home():
    """Landing page - redirect to login if not authenticated"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    """CKD prediction form"""
    return render_template('index.html', doctor=current_user)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        patient_name = request.form.get('patient_name', 'Unknown').strip()

        # 24 Features (model expects 24)
        features = np.array([[

            float(request.form['age']),
            float(request.form['bp']),
            float(request.form['sg']),
            float(request.form['al']),
            float(request.form['su']),

            0,  # rbc
            0,  # pc
            0,  # pcc
            0,  # ba

            float(request.form['bgr']),
            float(request.form['bu']),
            float(request.form['sc']),
            float(request.form['sod']),
            float(request.form['pot']),
            float(request.form['hemo']),

            float(request.form['pcv']),
            float(request.form['wbcc']),
            float(request.form['rbcc']),

            float(request.form['hypertension']),

            0,  # dm
            0,  # cad
            0,  # appet
            0,  # pe
            0   # ane

        ]])

        # Medical rule safety check
        if (
            features[0][7] <= 1.2 and
            features[0][6] <= 40 and
            features[0][10] >= 13 and
            features[0][2] >= 1.020 and
            features[0][3] == 0 and
            features[0][18] == 0
        ):
            result_text = "No CKD Detected"
            confidence = None
            is_ckd = False

        else:
            scaled = scaler.transform(features)

            ae_feat = encoder.predict(scaled)
            tt_feat = tabtransformer.predict(scaled)

            meta_feat = np.concatenate([ae_feat, tt_feat], axis=1)
            meta_feat = meta_scaler.transform(meta_feat)

            prob = meta_model.predict_proba(meta_feat)[0][1]

            confidence = round(prob * 100, 2)

            if prob >= THRESHOLD:
                result_text = "CKD Detected"
                is_ckd = True
            else:
                result_text = "No CKD Detected"
                is_ckd = False

        prediction = PredictionResult(
            doctor_id=current_user.id,
            patient_name=patient_name,
            age=float(request.form['age']),
            bp=float(request.form['bp']),
            sg=float(request.form['sg']),
            al=float(request.form['al']),
            su=float(request.form['su']),
            bgr=float(request.form['bgr']),
            bu=float(request.form['bu']),
            sc=float(request.form['sc']),
            sod=float(request.form['sod']),
            pot=float(request.form['pot']),
            hemo=float(request.form['hemo']),
            pcv=float(request.form['pcv']),
            wbcc=float(request.form['wbcc']),
            rbcc=float(request.form['rbcc']),
            hypertension=int(request.form['hypertension']),
            prediction=result_text,
            confidence=confidence
        )

        db.session.add(prediction)
        db.session.commit()

        return render_template(
            'result.html',
            result=result_text,
            confidence=confidence,
            patient_name=patient_name,
            doctor_name=current_user.full_name,
            is_ckd=is_ckd
        )

    except Exception as e:
        return render_template(
            'result.html',
            result=f"Error: {str(e)}",
            patient_name="Unknown",
            doctor_name=current_user.full_name,
            is_ckd=False,
            error=True
        )

if __name__ == '__main__':
    app.run(debug=True)
