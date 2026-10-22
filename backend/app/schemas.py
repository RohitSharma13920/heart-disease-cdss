from typing import Optional
from pydantic import BaseModel, Field, EmailStr

class AuthRequest(BaseModel):
    email: str = Field(..., description="Valid clinician or user email address")
    password: str = Field(..., min_length=4, description="User password")
    full_name: Optional[str] = Field("Dr. Clinician", description="Full name of user")
    role: Optional[str] = Field("Cardiologist", description="User role")

class PatientInput(BaseModel):
    patient_name: Optional[str] = Field("Anonymous Patient", max_length=100)
    user_email: Optional[str] = Field("Guest")
    
    # Strict range validation for clinical parameters
    age: int = Field(..., ge=18, le=110, description="Age in years (18-110)")
    sex: str = Field(..., description="Male or Female")
    bmi: float = Field(..., ge=10.0, le=65.0, description="Body Mass Index (10.0-65.0)")
    resting_bp: int = Field(..., ge=70, le=240, description="Resting Blood Pressure mm Hg")
    cholesterol: int = Field(..., ge=100, le=600, description="Serum Cholesterol mg/dl")
    max_heart_rate: int = Field(..., ge=60, le=230, description="Maximum Heart Rate achieved")
    
    smoking_status: Optional[str] = Field("Never", description="Never, Former, Current")
    alcohol_intake: Optional[str] = Field("Non-drinker", description="Non-drinker, Moderate, Heavy")
    physical_activity: Optional[str] = Field("Moderate", description="Sedentary, Light, Moderate, Active")
    diet_quality: Optional[int] = Field(6, ge=1, le=10)
    sleep_hours: Optional[float] = Field(7.0, ge=2.0, le=14.0)
    stress_level: Optional[int] = Field(5, ge=1, le=10)
    
    diabetes: int = Field(0, ge=0, le=1, description="0 = No, 1 = Yes")
    hypertension: int = Field(0, ge=0, le=1, description="0 = No, 1 = Yes")
    family_history: Optional[int] = Field(0, ge=0, le=1)
    previous_heart_event: Optional[int] = Field(0, ge=0, le=1)
    fasting_blood_sugar: Optional[int] = Field(0, ge=0, le=1)
    resting_ecg: Optional[str] = Field("Normal", description="Normal, ST-T Abnormality, LVH")
    exercise_angina: Optional[int] = Field(0, ge=0, le=1)
    st_depression: Optional[float] = Field(0.5, ge=0.0, le=8.0)
    st_slope: Optional[str] = Field("Flat", description="Upsloping, Flat, Downsloping")
    avg_resting_hr: Optional[float] = Field(72.0, ge=40.0, le=150.0)
    hrv_score: Optional[float] = Field(45.0, ge=5.0, le=120.0)
    clinical_notes: Optional[str] = Field("Routine cardiac risk screening.")

class NoteUpdateRequest(BaseModel):
    clinical_notes: str = Field(..., min_length=2, max_length=500)
    patient_name: Optional[str] = None