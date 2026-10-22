from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(100), nullable=False)
    email = Column(String(120), unique=True, index=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    role = Column(String(50), default="Clinician")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # One-to-Many Relationship: One user can have multiple prediction logs
    predictions = relationship(
        "PredictionLog",
        back_populates="owner",
        cascade="all, delete-orphan"
    )


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    user_email = Column(String(120), default="Guest")
    patient_name = Column(String(100), default="Anonymous Patient")
    
    # Clinical Input Features
    age = Column(Integer, nullable=False)
    sex = Column(String(20), nullable=False)
    bmi = Column(Float, nullable=False)
    resting_bp = Column(Integer, nullable=False)
    cholesterol = Column(Integer, nullable=False)
    max_heart_rate = Column(Integer, nullable=False)
    smoking_status = Column(String(30), nullable=False)
    diabetes = Column(Integer, nullable=False)
    hypertension = Column(Integer, nullable=False)

    # Multi-Model Outputs (ML + DL Comparison)
    rf_prediction = Column(Integer, nullable=False)
    rf_probability = Column(Float, nullable=False)
    lr_probability = Column(Float, nullable=False)
    dl_probability = Column(Float, nullable=False)
    risk_category = Column(String(50), nullable=False)
    clinical_notes = Column(Text, default="Standard automated diagnostic evaluation.")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    owner = relationship("User", back_populates="predictions")