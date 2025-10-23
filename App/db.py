from app.config import settings
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Create database engine only if database_url is provided
if hasattr(settings, 'database_url') and settings.database_url:
    engine = create_engine(settings.database_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()
else:
    engine = None
    SessionLocal = None
    Base = None
    print("Running without database - DATABASE_URL not configured")

def get_db():
    """Database dependency for FastAPI (if using FastAPI)"""
    if SessionLocal is None:
        raise RuntimeError("Database not configured")
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
