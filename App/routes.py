from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from ..db import SessionLocal
from ..models import Lead, Base
from ..agents.rag_agent import build_rag, answer_query
router = APIRouter()
class LeadIn(BaseModel):
name: str | None
email: str | None
phone: str | None
source: str | None
@router.post("/webhook/lead")
async def webhook_lead(payload: LeadIn):
# Very small example: insert lead into DB
db = SessionLocal()
lead = Lead(name=payload.name, email=payload.email, phone=payload.phone,
source=payload.source)
db.add(lead)
db.commit()
db.refresh(lead)
return {"status": "ok", "lead_id": lead.id}
class QueryIn(BaseModel):
q: str
@router.post("/ai/query")
async def ai_query(qin: QueryIn):
qa = build_rag()
resp = answer_query(qa, qin.q)
return {"answer": resp}
