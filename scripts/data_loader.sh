#!/usr/bin/env bash
# Simple script to create sample docs and ingest them
mkdir -p docs
cat > docs/followup.txt <<'EOF'
Recommended follow-up cadence for lawn-care leads in late spring: 1) initial
call/visit within 24 hours; 2) reminder SMS 3 days after; 3) follow-up call
at 10 days; 4) seasonal maintenance offer 30 days later.
EOF
# run ingestion inside container
docker compose exec api python -c "from app.ingest import ingest_folder;
ingest_folder('docs')"
