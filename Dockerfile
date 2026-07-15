# Semantix-Verify — MCP server image.
#
# Glama's harness builds THIS file (repo root), launches the container, and
# speaks JSON-RPC over stdio to enumerate and invoke tools. If this build
# fails, the Glama listing reports "quality — not tested" and the server is
# marked as not installable.
#
# libgomp1 is REQUIRED and is why this build previously failed: the [nli]
# extra pulls sentence-transformers -> torch, and torch links libgomp.so.1
# (OpenMP) at import time. python:*-slim does not ship it, so the warm-up
# below — and any real tool call — died with
# "libgomp.so.1: cannot open shared object file".
# docker/mcp.Dockerfile already had this fix; the root image never got it.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Unpinned: the previous ==0.2.0 pin silently froze this image behind PyPI.
RUN pip install "semantix-ai[nli,mcp]"

# Warm the cross-encoder weights into the image so the first
# verify_text_intent call doesn't block on a cold download inside Glama's
# invocation timeout. Requires libgomp1 above.
RUN python -c "from semantix.judges.nli import NLIJudge; NLIJudge()"

# MCP speaks stdio. No ports to EXPOSE.
CMD ["python", "-m", "semantix.mcp.server"]
