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
from semantix.composite import AllOf, AnyOf
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
    # Streaming
    "StreamCollector",
]

# QuantizedNLIJudge requires onnxruntime — optional export
try:
    from semantix.judges.quantized_nli import QuantizedNLIJudge

    __all__.append("QuantizedNLIJudge")
except ImportError:
    pass

__version__ = "0.1.9"
__author__ = "Akhona Eland"
