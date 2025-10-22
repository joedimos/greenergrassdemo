from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
import datetime
Base = declarative_base()
class Lead(Base):
__tablename__ = "leads"
id = Column(Integer, primary_key=True, index=True)
name = Column(String(255), nullable=True)
email = Column(String(255), nullable=True)
phone = Column(String(50), nullable=True)
source = Column(String(255), nullable=True)
4
created_at = Column(DateTime, default=datetime.datetime.utcnow)
class AuditLog(Base):
__tablename__ = "audit_logs"
id = Column(Integer, primary_key=True, index=True)
event_type = Column(String(255))
payload = Column(Text)
created_at = Column(DateTime, default=datetime.datetime.utcnow)
