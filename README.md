# GreenerGrass Demo
A demo scaffold to run a lead automation + RAG-enabled AI agent on a single
EC2 instance using Docker Compose.
## Quick start (on an Ubuntu EC2)
1. Install Docker & Docker Compose.
2. Clone this repo.
3. Copy `.env.template` to `.env` and fill values (OPENAI_API_KEY, etc.).
1
4. `docker compose up --build -d`
5. Visit `http://<EC2_PUBLIC_IP>:8000/docs` for FastAPI Swagger UI.
## Components
- FastAPI app (HTTP endpoints for webhook + QA).
- Qdrant vector store (container) for RAG.
- Postgres (container) for leads and audit logs.
- Redis (container) for Celery (optional in this demo; simple worker provided
without Celery).
## Notes
- This is a demo scaffold
