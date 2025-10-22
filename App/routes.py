from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session
from typing import Optional, Dict, Any
from functools import lru_cache
from ..db import get_db
from ..models import Lead, AuditLog
from ..agents.rag_agent import build_rag, answer_query
import json

router = APIRouter()

# Pydantic models
class LeadIn(BaseModel):
    name: Optional[str] = Field(None, max_length=255)
    email: Optional[EmailStr] = None
    phone: Optional[str] = Field(None, max_length=50)
    source: Optional[str] = Field(None, max_length=255)

class LeadOut(BaseModel):
    status: str
    lead_id: int
    message: str = "Lead created successfully"

class QueryIn(BaseModel):
    q: str = Field(..., min_length=1, max_length=1000, description="Question to ask")

class QueryOut(BaseModel):
    answer: str
    sources: Optional[list] = None
    query: str

# Cache RAG chain (build once, reuse)
@lru_cache()
def get_rag_chain():
    """Cached RAG chain builder."""
    return build_rag()

# Background task for audit logging
def log_audit_event(db: Session, event_type: str, payload: Dict[str, Any]):
    """Log audit event in background."""
    try:
        audit = AuditLog(
            event_type=event_type,
            payload=json.dumps(payload)
        )
        db.add(audit)
        db.commit()
    except Exception as e:
        print(f"Audit log error: {e}")
        db.rollback()

# Routes
@router.post("/webhook/lead", response_model=LeadOut, status_code=201)
async def webhook_lead(
    payload: LeadIn,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Create a new lead from webhook data.
    
    - **name**: Lead's name (optional)
    - **email**: Lead's email (optional)
    - **phone**: Lead's phone number (optional)
    - **source**: Source of the lead (optional)
    """
    try:
        # Validate at least one field is provided
        if not any([payload.name, payload.email, payload.phone]):
            raise HTTPException(
                status_code=400,
                detail="At least one of name, email, or phone must be provided"
            )
        
        # Create lead
        lead = Lead(
            name=payload.name,
            email=payload.email,
            phone=payload.phone,
            source=payload.source
        )
        db.add(lead)
        db.commit()
        db.refresh(lead)
        
        # Log audit event in background
        background_tasks.add_task(
            log_audit_event,
            db,
            "lead_created",
            {"lead_id": lead.id, "source": payload.source}
        )
        
        return {
            "status": "ok",
            "lead_id": lead.id,
            "message": "Lead created successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create lead: {str(e)}"
        )

@router.post("/ai/query", response_model=QueryOut)
async def ai_query(qin: QueryIn, db: Session = Depends(get_db)):
    """
    Answer a question using RAG (Retrieval Augmented Generation).
    
    - **q**: Question to ask the AI system
    """
    try:
        qa_chain = get_rag_chain()
        resp = answer_query(qa_chain, qin.q)
        
        # Log query
        audit = AuditLog(
            event_type="ai_query",
            payload=json.dumps({"query": qin.q[:200]})
        )
        db.add(audit)
        db.commit()
        
        # Handle both string and dict responses
        if isinstance(resp, dict):
            return {
                "answer": resp.get("answer", ""),
                "sources": resp.get("sources", []),
                "query": qin.q
            }
        return {
            "answer": resp,
            "sources": None,
            "query": qin.q
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )

@router.get("/leads/{lead_id}")
async def get_lead(lead_id: int, db: Session = Depends(get_db)):
    """Get a lead by ID."""
    lead = db.query(Lead).filter(Lead.id == lead_id).first()
    if not lead:
        raise HTTPException(status_code=404, detail="Lead not found")
    return lead

@router.get("/leads")
async def list_leads(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List all leads with pagination."""
    leads = db.query(Lead).offset(skip).limit(limit).all()
    total = db.query(Lead).count()
    return {
        "total": total,
        "leads": leads,
        "skip": skip,
        "limit": limit
    }
