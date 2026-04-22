FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install "semantix-ai[nli,mcp]==0.2.0"

RUN python -c "from semantix.judges.nli import NLIJudge; NLIJudge()"

ENTRYPOINT ["python", "-m", "semantix.mcp.server"]
