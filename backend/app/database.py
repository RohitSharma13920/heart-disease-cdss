import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

# Primary MySQL Configuration (Update credentials if using local MySQL server)
MYSQL_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:password@localhost:3306/heart_disease_db"
)

# Fallback SQLite Configuration (Ensures zero downtime if MySQL service is offline)
SQLITE_URL = "sqlite:///./heart_disease_enterprise.db"

def create_resilient_engine():
    try:
        # Attempt connection to MySQL Server with a short timeout
        engine = create_engine(
            MYSQL_URL,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 2}
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("[DATABASE] Connected to MySQL Server successfully.")
        return engine, "MySQL (Production ORM)"
    except Exception:
        print("[DATABASE] MySQL server unreachable. Activated resilient SQLite ORM fallback.")
        engine = create_engine(
            SQLITE_URL,
            connect_args={"check_same_thread": False}
        )
        return engine, "SQLAlchemy ORM (Resilient Local Engine)"

engine, ACTIVE_DB_ENGINE = create_resilient_engine()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency injection for FastAPI routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()