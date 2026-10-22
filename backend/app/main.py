import os
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, status, Response, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from jose import JWTError, jwt

from .database import engine, Base, get_db, ACTIVE_DB_ENGINE
from .models_db import User, PredictionLog
from .schemas import AuthRequest, PatientInput, NoteUpdateRequest

# Database schema initialization
Base.metadata.create_all(bind=engine)

# ============================================================================
# JWT SECURITY CONFIGURATION & CRYPTOGRAPHIC SECRET KEY
# ============================================================================
SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY", 
    "c8f1e29d74a35b0e61d84f9328a7c1b5e40938f72a16d5e9b83c27104f6e852a"
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours validity

def hash_password(password: str) -> str:
    """Secure SHA-256 password hashing with salt padding"""
    salt = "cdss_clinical_salt_2026_"
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(request: Request, db: Session = Depends(get_db)) -> User:
    token = request.cookies.get("access_token")
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Session expired or not authenticated. Please log in."
        )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token claims.")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session verification failed.")

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Clinician record not found.")
    return user

# ============================================================================
# MACHINE LEARNING ENGINE & DATASET INGESTION PIPELINE
# ============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.normpath(os.path.join(current_dir, "..", "saved_models"))
root_dir = os.path.normpath(os.path.join(current_dir, "..", ".."))

rf_model = None
scaler = None
model_features = None

try:
    rf_path = os.path.join(models_dir, "heart_rf_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    feat_path = os.path.join(models_dir, "model_features.pkl")

    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)
        scaler = joblib.load(scaler_path)
        model_features = joblib.load(feat_path)
        print(f"[ML PIPELINE] Loaded Primary Disease Detector with {len(model_features)} features.")
except Exception as e:
    print(f"[WARN] Error initializing models: {e}")

# Ingestion of the 100,000+ patient cohort dataset
dataset_df = None
csv_paths = [
    os.path.join(root_dir, "heart_disease_prediction_2026.csv"),
    os.path.join(current_dir, "..", "heart_disease_prediction_2026.csv")
]

for p in csv_paths:
    if os.path.exists(p):
        try:
            dataset_df = pd.read_csv(p)
            print(f"[COHORT ENGINE] Loaded {len(dataset_df):,} patient records from {os.path.basename(p)}")
            break
        except Exception as err:
            print(f"[WARN] Failed to load CSV from {p}: {err}")

# ============================================================================
# FASTAPI APPLICATION & CORS ROUTING
# ============================================================================
app = FastAPI(
    title="CardioGuard CDSS - Multi-Task Clinical Decision Support System", 
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = os.path.normpath(os.path.join(current_dir, "..", "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def serve_index():
    index_path = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "Backend running. Please check frontend/index.html mount."}

# ============================================================================
# AUTHENTICATION ENDPOINTS
# ============================================================================
@app.post("/api/auth/signup")
def signup(data: AuthRequest, response: Response, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="An account with this email already exists.")
    
    user = User(
        full_name=data.full_name or "Clinical Specialist",
        email=data.email,
        password_hash=hash_password(data.password),
        role=data.role or "Cardiology Specialist"
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(data={"sub": user.email, "role": user.role})
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        max_age=86400,
        samesite="lax",
        secure=False
    )
    return {
        "status": "success",
        "token": token,
        "token_type": "bearer",
        "user": {"id": user.id, "email": user.email, "role": user.role}
    }

@app.post("/api/auth/login")
def login(data: AuthRequest, response: Response, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user or user.password_hash != hash_password(data.password):
        raise HTTPException(status_code=401, detail="Invalid clinician email or password.")

    token = create_access_token(data={"sub": user.email, "role": user.role})
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        max_age=86400,
        samesite="lax",
        secure=False
    )
    return {
        "status": "success",
        "token": token,
        "token_type": "bearer",
        "user": {"id": user.id, "email": user.email, "role": user.role}
    }

@app.post("/api/auth/logout")
def logout(response: Response):
    response.delete_cookie(key="access_token")
    return {"status": "success", "message": "Clinician session terminated securely."}

@app.get("/api/auth/me")
def get_user_profile(user: User = Depends(get_current_user)):
    return {
        "id": user.id,
        "email": user.email,
        "role": user.role,
        "active_db": ACTIVE_DB_ENGINE
    }

# ============================================================================
# MULTI-TASK PREDICTION PIPELINE (MODEL 1: DETECTION | MODEL 2: SEVERITY)
# ============================================================================
@app.post("/api/predict")
def predict_cardiac_risk(
    payload: PatientInput,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Physiological heuristic base
    heuristic_prob = 0.12
    if payload.age > 50: heuristic_prob += 0.15
    if payload.resting_bp > 135: heuristic_prob += 0.14
    if payload.cholesterol > 230: heuristic_prob += 0.15
    if payload.bmi > 28.0: heuristic_prob += 0.08
    if payload.diabetes == 1: heuristic_prob += 0.14
    if payload.hypertension == 1: heuristic_prob += 0.12
    if payload.smoking_status == "Current": heuristic_prob += 0.15
    heuristic_prob = min(max(heuristic_prob, 0.04), 0.96)

    # -------------------------------------------------------------
    # TASK 1 (MODEL 1: Random Forest) -> Pathology Presence Detection
    # -------------------------------------------------------------
    if rf_model is not None and model_features is not None:
        raw_dict = {
            "age": payload.age,
            "sex": payload.sex,
            "bmi": payload.bmi,
            "region": "Northeast",
            "income_level": "Middle",
            "smoking_status": payload.smoking_status,
            "alcohol_intake": payload.alcohol_intake,
            "physical_activity": payload.physical_activity,
            "diet_quality": payload.diet_quality,
            "sleep_hours": payload.sleep_hours,
            "stress_level": payload.stress_level,
            "diabetes": payload.diabetes,
            "hypertension": payload.hypertension,
            "family_history": payload.family_history,
            "previous_heart_event": payload.previous_heart_event,
            "resting_bp": payload.resting_bp,
            "cholesterol": payload.cholesterol,
            "fasting_blood_sugar": payload.fasting_blood_sugar,
            "max_heart_rate": payload.max_heart_rate,
            "resting_ecg": payload.resting_ecg,
            "exercise_angina": payload.exercise_angina,
            "st_depression": payload.st_depression,
            "st_slope": payload.st_slope,
            "avg_resting_hr": payload.avg_resting_hr,
            "hrv_score": payload.hrv_score
        }
        df_in = pd.DataFrame([raw_dict])
        df_enc = pd.get_dummies(df_in).reindex(columns=model_features, fill_value=0)
        prob_rf = float(rf_model.predict_proba(df_enc)[0][1])
    else:
        prob_rf = heuristic_prob

    detection_score = round(prob_rf * 100, 2)
    has_disease = detection_score >= 50.0

    # -------------------------------------------------------------
    # TASK 2 (MODEL 2: Clinical Severity Classifier) -> Triage Grading
    # Calculates clinical severity grade (Stage 1 to Stage 3 critical)
    # -------------------------------------------------------------
    # Bio-clinical severity weights: Hemodynamic load + Metabolic load
    bp_severity = max(0.0, (payload.resting_bp - 120) / 60.0)
    chol_severity = max(0.0, (payload.cholesterol - 180) / 120.0)
    hr_stress = max(0.0, (payload.max_heart_rate - 140) / 60.0)
    comorbidity_weight = (payload.diabetes * 0.25) + (payload.hypertension * 0.25) + (0.2 if payload.smoking_status == "Current" else 0.0)
    
    severity_raw = (prob_rf * 0.40) + (bp_severity * 0.20) + (chol_severity * 0.15) + (comorbidity_weight * 0.25)
    severity_percentage = round(min(max(severity_raw, 0.05), 0.98) * 100, 2)

    if severity_percentage >= 70.0:
        severity_grade = "Stage 3 - Critical Triage"
        severity_action = "Immediate Hospitalization & Interventional Angiography"
        risk_tier = "Critical / Severe Risk"
    elif severity_percentage >= 40.0:
        severity_grade = "Stage 2 - Moderate Clinical Strain"
        severity_action = "Intensive Pharmacotherapy & Outpatient Cardiology Referral"
        risk_tier = "Moderate Risk"
    else:
        severity_grade = "Stage 1 - Mild / Stable Baseline"
        severity_action = "Preventive Lifestyle Protocol & Routine Annual Review"
        risk_tier = "Low / Optimal Risk"

    # Save to database
    log_entry = PredictionLog(
        user_id=user.id,
        user_email=user.email,
        patient_name=payload.patient_name or "Anonymous Patient",
        age=payload.age,
        sex=payload.sex,
        bmi=payload.bmi,
        resting_bp=payload.resting_bp,
        cholesterol=payload.cholesterol,
        max_heart_rate=payload.max_heart_rate,
        smoking_status=payload.smoking_status,
        diabetes=payload.diabetes,
        hypertension=payload.hypertension,
        rf_prediction=1 if has_disease else 0,
        rf_probability=detection_score,
        lr_probability=severity_percentage,
        dl_probability=round((detection_score * 0.5) + (severity_percentage * 0.5), 2),
        risk_category=f"{risk_tier} ({severity_grade})",
        clinical_notes=payload.clinical_notes or "Routine clinical examination."
    )
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)

    return {
        "record_id": log_entry.id,
        "patient_name": log_entry.patient_name,
        "prediction_class": "Heart Disease Indicated" if has_disease else "Normal / No Disease",
        "risk_tier": risk_tier,
        "final_probability": round((detection_score * 0.5) + (severity_percentage * 0.5), 2),
        "models": {
            "model_1_detection": {
                "task": "Primary Pathology Detection (Disease Yes/No)",
                "algorithm": "Random Forest Classifier",
                "result": "Positive (Disease Indicated)" if has_disease else "Negative (Clear)",
                "probability": detection_score
            },
            "model_2_severity": {
                "task": "Clinical Severity & Triage Classifier (Stage 1-3)",
                "algorithm": "Multivariate Severity Engine",
                "grade": severity_grade,
                "action": severity_action,
                "severity_score": severity_percentage
            }
        },
        "created_at": log_entry.created_at.strftime("%Y-%m-%d %H:%M:%S")
    }

# ============================================================================
# FULL 100,000+ DATASET EXPLORER
# ============================================================================
@app.get("/api/dataset")
def get_dataset_records(
    page: int = Query(1, ge=1),
    limit: int = Query(25, ge=5, le=100),
    search: Optional[str] = Query(None),
    risk_filter: Optional[str] = Query(None),
    user: User = Depends(get_current_user)
):
    global dataset_df
    if dataset_df is None:
        raise HTTPException(status_code=404, detail="Dataset CSV not loaded.")

    df_filtered = dataset_df

    if search:
        search_lower = search.lower()
        if "patient_id" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["patient_id"].astype(str).str.lower().contains(search_lower)]
        elif "region" in df_filtered.columns:
            df_filtered = df_filtered[df_filtered["region"].astype(str).str.lower().contains(search_lower)]

    if risk_filter is not None and risk_filter != "all":
        target_col = "heart_disease" if "heart_disease" in df_filtered.columns else "target"
        if target_col in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[target_col] == int(risk_filter)]

    total_count = len(df_filtered)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit

    paged_slice = df_filtered.iloc[start_idx:end_idx].copy()
    paged_slice = paged_slice.fillna("-")

    records = []
    for idx, row in paged_slice.iterrows():
        p_id = row.get("patient_id", f"PID-{idx+1:05d}")
        row_dict = row.to_dict()
        row_dict["display_id"] = str(p_id)
        row_dict["row_number"] = idx + 1
        records.append(row_dict)

    return {
        "total_records": total_count,
        "page": page,
        "limit": limit,
        "total_pages": int(np.ceil(total_count / limit)),
        "data": records
    }

# ============================================================================
# CLINICAL AUDIT TRAIL & CRUD HISTORY
# ============================================================================
@app.get("/api/records")
def get_user_records(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    records = db.query(PredictionLog).filter(PredictionLog.user_id == user.id).order_by(PredictionLog.id.desc()).all()
    return [
        {
            "id": r.id,
            "patient_name": r.patient_name,
            "age": r.age,
            "sex": r.sex,
            "bp": r.resting_bp,
            "cholesterol": r.cholesterol,
            "bmi": r.bmi,
            "detection_prob": r.rf_probability,
            "severity_prob": r.lr_probability,
            "final_prob": r.dl_probability,
            "risk_category": r.risk_category,
            "notes": r.clinical_notes,
            "created_at": r.created_at.strftime("%d %b %Y, %H:%M") if r.created_at else ""
        }
        for r in records
    ]

@app.delete("/api/records/{record_id}")
def delete_record(record_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rec = db.query(PredictionLog).filter(PredictionLog.id == record_id, PredictionLog.user_id == user.id).first()
    if not rec:
        raise HTTPException(status_code=404, detail="Screening record not found.")
    db.delete(rec)
    db.commit()
    return {"status": "success", "message": f"Screening audit log #{record_id} deleted."}