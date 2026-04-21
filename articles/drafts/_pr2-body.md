Adds semantix-ai under LLM Evaluation (inside the "other evaluation frameworks" details block, alongside Giskard / LangSmith / Ragas).

semantix-ai is an open-source (MIT) Python library for validating LLM outputs against plain-English semantic intents using a local quantized NLI model (~25 MB ONNX, ~15 ms per call, deterministic, no API key).

- Drop-in reward/metric functions for DSPy (`semantic_reward`, `semantic_metric` — compatible with `BestOfN`, `Refine`, `Evaluate`, MIPROv2).
- pytest integration via [`pytest-semantix`](https://github.com/labrat-akhona/pytest-semantix).
- Hash-chained JSON-LD audit receipts for every validation — tamper-evident.
- Integrations: LangChain, Pydantic AI, Guardrails, Instructor, MCP.

Reproducible benchmarks comparing against LLM-judge baselines: https://github.com/labrat-akhona/semantix-ai/tree/master/benchmarks

- Repo: https://github.com/labrat-akhona/semantix-ai
- PyPI: https://pypi.org/project/semantix-ai/
- Docs: https://labrat-akhona.github.io/semantix-ai/
- License: MIT
