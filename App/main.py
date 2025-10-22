from fastapi import FastAPI
from .api.routes import router
from .models import Base
from .db import engine

# Create DB tables on startup for demo
Base.metadata.create_all(bind=engine)

app = FastAPI(title="GreenerGrass Demo")
app.include_router(router)

@app.get("/health")
async def health():
    return {"status": "ok"}
