"""Pre-built Intent presets for POPIA (Protection of Personal Information Act).

Each preset is a module-level :class:`~semantix.intent.Intent` instance anchored
to one of the POPIA concept labels in :meth:`POPIAJudge.clauses`. Use with
:class:`~semantix.judges.popia.POPIAJudge`.

These presets encode an engineering reading of POPIA for the purpose of
automated output validation. They are not legal advice. Consult your DPIA and
legal counsel for compliance determinations.
"""

from __future__ import annotations

from semantix.intent import Intent

__all__ = [
    "POPIA_CONSENT",
    "POPIA_MINIMALITY",
    "POPIA_SECURITY",
    "POPIA_BREACH",
    "POPIA_CROSS_BORDER",
    "POPIA_PROCESSING",
    "POPIA_DATA_SUBJECT_RIGHTS",
]


POPIA_CONSENT = Intent(
    description=(
        "The output obtains or confirms explicit, informed, voluntary consent "
        "from the data subject before processing their personal information."
    ),
    clause="POPIA consent",
    negate=False,
)

POPIA_MINIMALITY = Intent(
    description=(
        "The output processes personal information only to the extent necessary "
        "for a specific, explicit, lawful purpose and does not use it for "
        "incompatible secondary purposes."
    ),
    clause="POPIA minimality / purpose limitation",
    negate=False,
)

POPIA_SECURITY = Intent(
    description=(
        "The output describes appropriate, reasonable technical and "
        "organisational security safeguards protecting personal information "
        "against loss, damage, unauthorised access, or unlawful processing."
    ),
    clause="POPIA security safeguards",
    negate=False,
    threshold=0.85,
)

POPIA_BREACH = Intent(
    description=(
        "The output delays, minimises, or fails to notify affected data subjects "
        "and the Information Regulator after a security compromise involving "
        "personal information."
    ),
    clause="POPIA breach notification",
    negate=True,
)

POPIA_CROSS_BORDER = Intent(
    description=(
        "The output transfers personal information to a recipient in a foreign "
        "country without ensuring adequate protection equivalent to POPIA, "
        "data subject consent, or another lawful basis for the transfer."
    ),
    clause="POPIA cross-border transfers",
    negate=False,
)

POPIA_PROCESSING = Intent(
    description=(
        "The output processes personal information lawfully, in a reasonable "
        "manner that does not infringe the privacy of the data subject."
    ),
    clause="POPIA general processing",
    negate=False,
)

POPIA_DATA_SUBJECT_RIGHTS = Intent(
    description=(
        "The output respects the data subject's rights to access, correct, "
        "delete, or object to the processing of their personal information "
        "and responds to such requests within a reasonable time."
    ),
    clause="POPIA data subject rights",
    negate=False,
)
