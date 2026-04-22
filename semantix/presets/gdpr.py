"""Pre-built Intent presets for the EU General Data Protection Regulation (GDPR).

Each preset is a module-level :class:`~semantix.intent.Intent` instance anchored
to one of seven canonical GDPR clauses. Use with
:class:`~semantix.judges.gdpr.GDPRJudge`.

These presets encode an engineering reading of GDPR for the purpose of
automated output validation. They are not legal advice.

**Status:** scaffold shipped in v0.2.1. The backing fine-tuned model
(`labrat-aiko/nli-gdpr-v1`) is not yet published — see
`docs/superpowers/specs/2026-04-22-gdpr-v0-scaffold.md` for the roadmap.
Until the fine-tune is published, `GDPRJudge` will fall back to the
generic `QuantizedNLIJudge` weights and performance will be closer to
stock NLI than to a clause-specialised judge.
"""

from __future__ import annotations

from semantix.intent import Intent

__all__ = [
    "GDPR_CONSENT",
    "GDPR_MINIMALITY",
    "GDPR_SECURITY",
    "GDPR_BREACH",
    "GDPR_TRANSFERS",
    "GDPR_PROCESSING",
    "GDPR_DATA_SUBJECT_RIGHTS",
]


GDPR_CONSENT = Intent(
    description=(
        "The output obtains specific, freely-given, informed, and unambiguous "
        "consent from the data subject for each distinct processing purpose, "
        "in line with Articles 6(1)(a) and 7 GDPR."
    ),
    clause="GDPR consent",
    negate=False,
)

GDPR_MINIMALITY = Intent(
    description=(
        "The output limits personal-data collection to what is adequate, "
        "relevant, and necessary for the specific purpose, in line with "
        "Article 5(1)(c) GDPR."
    ),
    clause="GDPR minimality",
    negate=False,
)

GDPR_SECURITY = Intent(
    description=(
        "The output implements appropriate technical and organisational "
        "measures to ensure a level of security proportionate to the risk, "
        "in line with Article 32 GDPR."
    ),
    clause="GDPR security",
    negate=False,
    threshold=0.85,
)

GDPR_BREACH = Intent(
    description=(
        "The output delays, minimises, or fails to notify the supervisory "
        "authority within 72 hours of becoming aware of a personal-data "
        "breach, or fails to communicate to affected data subjects where "
        "the breach is likely to result in a high risk to their rights, "
        "in breach of Articles 33 and 34 GDPR."
    ),
    clause="GDPR breach notification",
    negate=True,
)

GDPR_TRANSFERS = Intent(
    description=(
        "The output transfers personal data to a third country or international "
        "organisation without an adequacy decision, Standard Contractual "
        "Clauses, Binding Corporate Rules, or another lawful basis under "
        "Chapter V of the GDPR."
    ),
    clause="GDPR cross-border transfers",
    negate=False,
)

GDPR_PROCESSING = Intent(
    description=(
        "The output processes personal data lawfully, fairly, and in a "
        "transparent manner towards the data subject, in line with "
        "Article 5(1)(a) GDPR."
    ),
    clause="GDPR general processing",
    negate=False,
)

GDPR_DATA_SUBJECT_RIGHTS = Intent(
    description=(
        "The output respects and enables the data subject's rights of access, "
        "rectification, erasure, restriction, portability, objection, and "
        "safeguards against automated decision-making, in line with "
        "Articles 15–22 GDPR."
    ),
    clause="GDPR data subject rights",
    negate=False,
)
