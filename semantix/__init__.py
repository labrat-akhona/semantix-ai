"""Semantix — A Semantic Type System for AI outputs.

Define *what* your LLM output should mean, not just what shape it has.

Quick start
-----------
>>> from semantix import Intent, validate_intent, SemanticIntentError
>>>
>>> class ProfessionalDecline(Intent):
...     \"\"\"The text must politely decline an invitation without being
...     rude or aggressive.\"\"\"
>>>
>>> @validate_intent
... def decline_invite(event: str) -> ProfessionalDecline:
...     return call_my_llm(event)  # returns a plain string
"""

from semantix.audit.engine import AuditEngine
from semantix.composite import AllOf, AnyOf, Not
from semantix.decorator import get_last_failure, validate_intent
from semantix.exceptions import SemanticIntentError
from semantix.intent import Intent
from semantix.judges import Judge, Verdict
from semantix.judges.caching import CachingJudge
from semantix.judges.embedding import EmbeddingJudge
from semantix.judges.forensic import ForensicJudge
from semantix.judges.llm import LLMJudge
from semantix.judges.nli import NLIJudge
from semantix.streaming import StreamCollector
from semantix.testing import assert_semantic

__all__ = [
    # Core
    "Intent",
    "SemanticIntentError",
    "validate_intent",
    "get_last_failure",
    "assert_semantic",
    # Judges
    "Judge",
    "Verdict",
    "EmbeddingJudge",
    "NLIJudge",
    "LLMJudge",
    "CachingJudge",
    "ForensicJudge",
    # Audit
    "AuditEngine",
    # Composite
    "AllOf",
    "AnyOf",
    "Not",
    # Streaming
    "StreamCollector",
]

# QuantizedNLIJudge requires onnxruntime — optional export
try:
    from semantix.judges.quantized_nli import QuantizedNLIJudge

    __all__.append("QuantizedNLIJudge")
except ImportError:
    pass

# POPIAJudge shares the quantized-NLI deps; available when [popia] extras installed
try:
    from semantix.judges.popia import POPIAJudge  # noqa: F401

    __all__.append("POPIAJudge")
except ImportError:
    pass

# GDPRJudge shares the quantized-NLI deps; available when [gdpr] extras installed.
# The fine-tuned weights are not yet published — GDPRJudge falls back to the
# POPIA weights at runtime with a warning until labrat-aiko/nli-gdpr-v1 ships.
try:
    from semantix.judges.gdpr import GDPRJudge  # noqa: F401

    __all__.append("GDPRJudge")
except ImportError:
    pass

__version__ = "0.2.0"
__author__ = "Akhona Eland"
