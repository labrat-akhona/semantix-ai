# Dockerfile for the Semantix-Verify MCP server
#
# Published for Glama's automated safety/quality harness. Glama runs this image,
# talks to the MCP server over stdio, and verifies tool discovery + invocation
# before the listing becomes searchable.
#
# Matches the README install command: pip install "semantix-ai[mcp,nli]"
# The [nli] extra pulls sentence-transformers (torch-based) for the generic
# NLI judge used by the verify_text_intent tool. First invocation downloads
# the cross-encoder weights (~90MB) from Hugging Face.

FROM python:3.11-slim

# libgomp1 is required at runtime by torch/onnxruntime for OpenMP-based kernels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "semantix-ai[mcp,nli]"

# MCP servers communicate via stdio. No ports to EXPOSE.
# Glama's harness launches the container and pipes JSON-RPC over stdin/stdout.
CMD ["python", "-m", "semantix.mcp.server"]
