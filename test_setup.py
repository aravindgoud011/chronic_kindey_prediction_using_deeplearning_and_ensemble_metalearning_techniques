#!/usr/bin/env python3
"""Quick test to verify app setup"""

import sys
sys.path.insert(0, r'c:\Users\aravi\Desktop\CKD_Prediction_Project\app')

try:
    print("✓ Testing imports...")
    from app import app, db, Doctor, PredictionResult
    print("✓ Flask and SQLAlchemy imported successfully")
    
    # Test database creation
    with app.app_context():
        db.create_all()
        print("✓ Database tables created successfully")
        
        # Check existing doctors
        doctor_count = Doctor.query.count()
        print(f"✓ Existing doctors in database: {doctor_count}")
    
    print("\n✅ All checks passed! Your app is ready to run.")
    print("\nTo start the app, run:")
    print("  python app/app.py")
    print("\nThen visit:")
    print("  http://localhost:5000/register - to register a new doctor")
    print("  http://localhost:5000/login - to login")
    
except Exception as e:
    print(f"❌ Error during setup: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
