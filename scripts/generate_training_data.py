#!/usr/bin/env python3
"""
generate_training_data.py — Synthetic POPIA training-data pipeline for Semantix AI.

Implements four generation strategies:
  1. Evol-Instruct style complexity evolution
  2. Hard-negative generation (SyNeg-style)
  3. Scenario diversification (industry × company size × data type × action)
  4. Curriculum ordering (difficulty 1-5)

Reads seed data from data/*.jsonl and writes to data/popia_synthetic_v1.jsonl
with full metadata (difficulty, clause_ids, scenario_type, generation_method).

All generation is template-based (no LLM calls).

Usage:
    python scripts/generate_training_data.py [--seed-dir data] [--output data/popia_synthetic_v1.jsonl]
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import random
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Domain knowledge ─────────────────────────────────────────────────────────

POPIA_CLAUSES: dict[str, dict[str, Any]] = {
    "POPIA consent": {
        "sections": ["§11", "§12"],
        "short": "consent",
        "keywords": ["consent", "opt-in", "opt-out", "agreement", "freely given"],
    },
    "POPIA minimality / purpose limitation": {
        "sections": ["§10", "§13", "§14"],
        "short": "minimality",
        "keywords": ["minimality", "purpose limitation", "adequate", "relevant", "excessive", "retention"],
    },
    "POPIA security safeguards": {
        "sections": ["§19", "§20", "§21"],
        "short": "security",
        "keywords": ["security", "safeguards", "encryption", "access control", "breach prevention"],
    },
    "POPIA breach notification": {
        "sections": ["§22"],
        "short": "breach",
        "keywords": ["breach", "notification", "compromise", "unauthorised access", "incident"],
    },
    "POPIA cross-border transfers": {
        "sections": ["§72"],
        "short": "cross-border",
        "keywords": ["cross-border", "transfer", "international", "offshore", "foreign"],
    },
    "POPIA general processing": {
        "sections": ["§8", "§9"],
        "short": "general",
        "keywords": ["lawfulness", "processing", "conditions", "accountability", "privacy impact"],
    },
    "POPIA data subject rights": {
        "sections": ["§5", "§23", "§24", "§25"],
        "short": "dsr",
        "keywords": ["access", "correction", "deletion", "objection", "data subject"],
    },
    "POPIA special personal information": {
        "sections": ["§26", "§27", "§28", "§29", "§30", "§31", "§32"],
        "short": "special",
        "keywords": ["special personal", "race", "religion", "health", "biometric", "criminal"],
    },
    "POPIA children's information": {
        "sections": ["§34", "§35"],
        "short": "children",
        "keywords": ["child", "minor", "parental consent", "competent person", "under 18"],
    },
    "POPIA automated decision-making": {
        "sections": ["§71"],
        "short": "automated",
        "keywords": ["automated", "profiling", "algorithmic", "decision-making", "human review"],
    },
}

INDUSTRIES = [
    ("fintech", "a South African fintech startup"),
    ("healthcare", "a private hospital group in Gauteng"),
    ("education", "a public university in the Western Cape"),
    ("mining", "a Mpumalanga coal-mining company"),
    ("insurance", "a short-term insurer based in Sandton"),
    ("retail", "a national retail chain with 200 stores"),
    ("government", "a KZN metropolitan municipality"),
    ("telco", "a mobile network operator with 20 million subscribers"),
    ("legal", "a mid-size Johannesburg law firm"),
    ("logistics", "a cross-border freight and logistics company"),
    ("ngo", "a Cape Town-based NGO serving refugee communities"),
    ("agriculture", "a commercial farming operation in the Free State"),
]

COMPANY_SIZES = [
    ("startup", "a 12-person startup"),
    ("sme", "a 60-employee SME"),
    ("enterprise", "a JSE-listed enterprise with 5 000 staff"),
    ("multinational", "a multinational with South African operations"),
]

DATA_TYPES = [
    ("biometric", "fingerprint and facial-recognition data"),
    ("health", "patient medical records and chronic-condition data"),
    ("children", "personal information of learners under 18"),
    ("financial", "bank-account numbers, credit scores and transaction histories"),
    ("criminal", "criminal-record and vetting information"),
    ("employee", "employee HR and payroll records"),
    ("marketing", "marketing opt-in lists and behavioural-tracking cookies"),
    ("identity", "ID numbers, passport numbers and proof-of-address documents"),
]

ACTIONS = [
    ("collection", "collecting"),
    ("processing", "processing"),
    ("storage", "storing"),
    ("transfer", "transferring to a third party"),
    ("cross-border", "transferring outside South Africa"),
    ("breach", "suffering a breach involving"),
    ("deletion", "deleting"),
    ("retention", "retaining beyond the original period"),
    ("profiling", "using for automated profiling"),
    ("sharing", "sharing with a business partner"),
]

# ── Evol-Instruct templates ─────────────────────────────────────────────────

CONSTRAINT_TEMPLATES = [
    "What if the data subject is a minor AND the processing is for {action}?",
    "What if the responsible party is a {company_size} AND the data involved is {data_type}?",
    "Consider a scenario where {industry_desc} is {action_verb} {data_type_desc}. What additional POPIA obligations apply?",
    "What if the original consent was obtained before POPIA commenced, and the responsible party is now {action_verb} the data for a new purpose?",
    "What if the data subject has objected to the processing, but the responsible party argues legitimate interest?",
    "How does the analysis change if the {data_type_desc} is being processed by an operator (not the responsible party)?",
]

REASONING_TEMPLATES = [
    "Identify which POPIA sections apply to this scenario and explain step-by-step why each is relevant.",
    "Walk through the eight conditions for lawful processing and assess whether each is met.",
    "Explain which POPIA exemption, if any, the responsible party might rely on, and whether it would succeed.",
    "Analyse whether the responsible party would satisfy the 'reasonable measures' test in this context.",
    "Compare how this scenario would be assessed under POPIA versus GDPR — what are the key differences?",
]

MULTI_HOP_TEMPLATES = [
    (
        "{industry_desc} is {action_verb} {data_type_desc} of minors, and also transferring it to "
        "a processor in {foreign_country}. Which POPIA sections are engaged simultaneously?"
    ),
    (
        "A data breach at {industry_desc} exposed {data_type_desc}. The company delayed notification "
        "because law enforcement asked them to. Three months later, data subjects request access to "
        "all records about them. Trace the full chain of POPIA obligations."
    ),
    (
        "{industry_desc} uses automated decision-making on {data_type_desc} collected from children. "
        "Identify every applicable POPIA provision and explain how they interact."
    ),
    (
        "An employee at {industry_desc} discovers that {data_type_desc} is being processed without "
        "a lawful basis and blows the whistle. Map the relevant POPIA sections from collection "
        "through to enforcement."
    ),
]

EDGE_CASE_TEMPLATES = [
    (
        "{industry_desc} argues that because their servers are in South Africa, POPIA §72 does not "
        "apply — but their cloud provider replicates data to a foreign region. Is this a cross-border "
        "transfer?"
    ),
    (
        "A data subject asks {industry_desc} to delete all their data, but the company is legally "
        "required to retain certain records under FICA/the Tax Administration Act. How should the "
        "company respond?"
    ),
    (
        "{industry_desc} obtained consent for purpose A, then argues that purpose B is 'compatible'. "
        "Under what conditions is this acceptable under POPIA §15?"
    ),
    (
        "The Information Regulator issues an enforcement notice, but {industry_desc} claims the "
        "processing falls under the journalism exemption in §7. Is this defence available?"
    ),
    (
        "{industry_desc} processes {data_type_desc} for research purposes and claims the §27 "
        "exemption. What conditions must be met for this exemption to apply?"
    ),
]

FOREIGN_COUNTRIES = [
    "the United Kingdom", "India", "the United States", "Nigeria",
    "Mauritius", "the UAE", "Germany", "Kenya", "China", "Australia",
]

# ── Hard-negative templates ──────────────────────────────────────────────────

NEAR_MISS_MUTATIONS: list[dict[str, Any]] = [
    {
        "desc": "flip_timeframe",
        "transform": lambda p: p.replace("the same day", "six months later")
                                .replace("within 21 days", "after eight months")
                                .replace("the next working day", "three quarters later")
                                .replace("within four hours", "four months later"),
        "label_flip": {"entailment": "contradiction"},
    },
    {
        "desc": "remove_safeguard",
        "transform": lambda p: p.replace("encrypted at rest with AES-256, ", "")
                                .replace("MFA-enforced and ", "")
                                .replace("role-based, ", "")
                                .replace("documented ", ""),
        "label_flip": {"entailment": "neutral"},
    },
    {
        "desc": "add_violation",
        "transform": lambda p: p + " However, we also use this data for undisclosed marketing purposes.",
        "label_flip": {"entailment": "contradiction"},
    },
    {
        "desc": "weaken_consent",
        "transform": lambda p: p.replace("signed", "verbally mentioned")
                                .replace("explicit", "implied")
                                .replace("written form", "casual conversation")
                                .replace("documented consent", "informal agreement"),
        "label_flip": {"entailment": "neutral"},
    },
]

SECTION_CONFUSION_PAIRS: list[dict[str, str]] = [
    {
        "wrong_clause": "POPIA consent",
        "correct_clause": "POPIA security safeguards",
        "hypothesis_template": "The responsible party has obtained valid consent for the processing.",
    },
    {
        "wrong_clause": "POPIA breach notification",
        "correct_clause": "POPIA security safeguards",
        "hypothesis_template": "The responsible party has notified the Regulator as required under §22.",
    },
    {
        "wrong_clause": "POPIA cross-border transfers",
        "correct_clause": "POPIA general processing",
        "hypothesis_template": "The cross-border transfer satisfies at least one of the §72 lawful bases.",
    },
    {
        "wrong_clause": "POPIA data subject rights",
        "correct_clause": "POPIA consent",
        "hypothesis_template": "The responsible party is honouring the data subject's §23 access right.",
    },
    {
        "wrong_clause": "POPIA automated decision-making",
        "correct_clause": "POPIA special personal information",
        "hypothesis_template": "The data subject has been notified of the automated decision under §71.",
    },
]

PARTIAL_COMPLIANCE_SUFFIXES = [
    " However, third parties who received the data were never informed of the change.",
    " However, no written record of the action was created for audit purposes.",
    " However, the data subject was never informed of the outcome.",
    " However, the privacy notice was not updated to reflect this new processing.",
    " However, the company did not notify the Information Regulator as required.",
    " However, the operator agreement does not include data-protection obligations.",
]

# ── Dataclass for generated examples ────────────────────────────────────────

@dataclass
class SyntheticExample:
    """A single generated training example with full provenance metadata."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    # NLI fields
    clause: str = ""
    premise: str = ""
    hypothesis: str = ""
    label: str = ""
    scenario: str = ""
    # Instruct fields (optional — only for instruct-format examples)
    messages: list[dict[str, str]] | None = None
    # Metadata
    difficulty: int = 1
    clause_ids: list[str] = field(default_factory=list)
    scenario_type: str = ""
    generation_method: str = ""
    source_id: str = ""
    tags: list[str] = field(default_factory=list)

    def to_nli_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "clause": self.clause,
            "premise": self.premise,
            "hypothesis": self.hypothesis,
            "label": self.label,
            "scenario": self.scenario,
            "difficulty": self.difficulty,
            "clause_ids": self.clause_ids,
            "scenario_type": self.scenario_type,
            "generation_method": self.generation_method,
            "source_id": self.source_id,
            "tags": self.tags,
        }

    def to_instruct_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "messages": self.messages,
            "difficulty": self.difficulty,
            "clause_ids": self.clause_ids,
            "scenario_type": self.scenario_type,
            "generation_method": self.generation_method,
            "source_id": self.source_id,
            "tags": self.tags,
        }

    def to_dict(self) -> dict[str, Any]:
        if self.messages is not None:
            return self.to_instruct_dict()
        return self.to_nli_dict()


# ── Seed loader ──────────────────────────────────────────────────────────────

def load_seeds(seed_dir: Path) -> tuple[list[dict], list[dict]]:
    """Load NLI seeds and instruct seeds from the data directory."""
    nli_seeds: list[dict] = []
    instruct_seeds: list[dict] = []

    nli_files = [
        "popia_seeds.jsonl",
        "popia_seeds_v2.jsonl",
        "popia_paraphrases.jsonl",
        "popia_paraphrases_v2.jsonl",
    ]
    instruct_files = [
        "popia_instruct.jsonl",
    ]

    for fname in nli_files:
        fpath = seed_dir / fname
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if "premise" in obj and "hypothesis" in obj:
                            obj.setdefault("_source_file", fname)
                            obj.setdefault("_source_idx", i)
                            nli_seeds.append(obj)
                    except json.JSONDecodeError:
                        log.warning("Skipping malformed line %d in %s", i, fname)
            log.info("Loaded %s: found entries so far: %d NLI", fname, len(nli_seeds))

    for fname in instruct_files:
        fpath = seed_dir / fname
        if fpath.exists():
            with open(fpath, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if "messages" in obj:
                            obj.setdefault("_source_file", fname)
                            obj.setdefault("_source_idx", i)
                            instruct_seeds.append(obj)
                    except json.JSONDecodeError:
                        log.warning("Skipping malformed line %d in %s", i, fname)
            log.info("Loaded %s: found entries so far: %d instruct", fname, len(instruct_seeds))

    return nli_seeds, instruct_seeds


# ── 1. Evol-Instruct complexity evolution ────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a careful South African data-protection assistant. Answer in clear, "
    "professional English with reference to the Protection of Personal Information "
    "Act, 2013 (POPIA). When citing sections, use the form 'POPIA §X'. Be specific, "
    "not vague. If a scenario is non-compliant, name the rule it breaks."
)


def evolve_with_constraints(
    nli_seeds: list[dict], rng: random.Random, count: int = 80
) -> list[SyntheticExample]:
    """Evolve simple seeds by injecting constraints, reasoning demands, and multi-hop."""
    results: list[SyntheticExample] = []

    for _ in range(count):
        seed = rng.choice(nli_seeds)
        clause = seed["clause"]
        clause_info = POPIA_CLAUSES.get(clause, {})
        industry_name, industry_desc = rng.choice(INDUSTRIES)
        _, company_size_desc = rng.choice(COMPANY_SIZES)
        data_type_name, data_type_desc = rng.choice(DATA_TYPES)
        action_name, action_verb = rng.choice(ACTIONS)

        strategy = rng.choice(["constraint", "reasoning", "multi_hop", "edge_case"])

        if strategy == "constraint":
            template = rng.choice(CONSTRAINT_TEMPLATES)
            user_q = template.format(
                action=action_name,
                company_size=company_size_desc,
                data_type=data_type_name,
                industry_desc=industry_desc,
                action_verb=action_verb,
                data_type_desc=data_type_desc,
            )
            base_scenario = seed["premise"]
            full_question = f"Given this scenario: \"{base_scenario}\"\n\n{user_q}"
            sections = clause_info.get("sections", [])
            difficulty = 2

            answer = (
                f"This scenario engages {', '.join(sections)} ({clause}). "
                f"The added constraint — {user_q.split('?')[0].lower().strip()}? — "
                f"means additional obligations arise. "
                f"Specifically, the responsible party must ensure compliance with "
                f"the conditions for lawful processing under the added circumstances."
            )

        elif strategy == "reasoning":
            reasoning_q = rng.choice(REASONING_TEMPLATES)
            base_scenario = seed["premise"]
            full_question = f"Scenario: \"{base_scenario}\"\n\n{reasoning_q}"
            sections = clause_info.get("sections", [])
            difficulty = 3

            answer = (
                f"Applying {clause} ({', '.join(sections)}) to this scenario:\n\n"
                f"Step 1: Identify the processing activity described.\n"
                f"Step 2: Determine the applicable POPIA condition(s).\n"
                f"Step 3: Assess whether the scenario satisfies each condition.\n"
                f"Step 4: Conclude whether the processing is compliant.\n\n"
                f"The scenario described — '{base_scenario[:80]}...' — "
                f"{'meets' if seed.get('label') == 'entailment' else 'does not meet'} "
                f"the requirements under {', '.join(sections)}."
            )

        elif strategy == "multi_hop":
            foreign_country = rng.choice(FOREIGN_COUNTRIES)
            template = rng.choice(MULTI_HOP_TEMPLATES)
            full_question = template.format(
                industry_desc=industry_desc,
                action_verb=action_verb,
                data_type_desc=data_type_desc,
                foreign_country=foreign_country,
            )
            # Multi-hop engages multiple clause areas
            other_clause = rng.choice(
                [c for c in POPIA_CLAUSES if c != clause]
            )
            other_info = POPIA_CLAUSES[other_clause]
            all_sections = clause_info.get("sections", []) + other_info.get("sections", [])
            sections = all_sections
            difficulty = 4

            answer = (
                f"This multi-faceted scenario engages at least two areas of POPIA:\n\n"
                f"1. {clause} ({', '.join(clause_info.get('sections', []))})\n"
                f"2. {other_clause} ({', '.join(other_info.get('sections', []))})\n\n"
                f"The responsible party — {industry_desc} — must satisfy the requirements "
                f"of each simultaneously. A failure under any one provision would render "
                f"the overall processing non-compliant."
            )

        else:  # edge_case
            template = rng.choice(EDGE_CASE_TEMPLATES)
            full_question = template.format(
                industry_desc=industry_desc,
                data_type_desc=data_type_desc,
            )
            sections = clause_info.get("sections", [])
            difficulty = 5

            answer = (
                f"This is an edge case under {clause}. The answer depends on the specific "
                f"facts: {full_question.split('?')[0].lower().strip()}? "
                f"Under {', '.join(sections)}, the responsible party would need to "
                f"demonstrate that the processing meets the applicable conditions. "
                f"The Information Regulator's guidance and case law would inform the analysis."
            )

        ex = SyntheticExample(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_question},
                {"role": "assistant", "content": answer},
            ],
            difficulty=difficulty,
            clause_ids=sections,
            scenario_type=f"{industry_name}-{action_name}",
            generation_method=f"evol-instruct-{strategy}",
            source_id=f"{seed.get('_source_file', 'unknown')}:{seed.get('_source_idx', -1)}",
            tags=[strategy, industry_name, data_type_name, action_name],
        )
        results.append(ex)

    return results


# ── 2. Hard-negative generation (SyNeg-style) ───────────────────────────────

def generate_hard_negatives(
    nli_seeds: list[dict], rng: random.Random, count: int = 100
) -> list[SyntheticExample]:
    """Generate near-miss, section-confusion, and partial-compliance negatives."""
    results: list[SyntheticExample] = []
    entailment_seeds = [s for s in nli_seeds if s.get("label") == "entailment"]
    all_clauses = list(POPIA_CLAUSES.keys())

    # ── Near-miss negatives ──────────────────────────────────────────────────
    near_miss_budget = count // 3
    for _ in range(near_miss_budget):
        seed = rng.choice(entailment_seeds)
        mutation = rng.choice(NEAR_MISS_MUTATIONS)
        new_premise = mutation["transform"](seed["premise"])

        # Only keep if the premise actually changed
        if new_premise == seed["premise"]:
            continue

        new_label = mutation["label_flip"].get(seed["label"], seed["label"])
        clause_info = POPIA_CLAUSES.get(seed["clause"], {})

        ex = SyntheticExample(
            clause=seed["clause"],
            premise=new_premise,
            hypothesis=seed["hypothesis"],
            label=new_label,
            scenario=seed.get("scenario", "") + "-near-miss",
            difficulty=3,
            clause_ids=clause_info.get("sections", []),
            scenario_type="hard-negative",
            generation_method=f"syneg-near-miss-{mutation['desc']}",
            source_id=f"{seed.get('_source_file', 'unknown')}:{seed.get('_source_idx', -1)}",
            tags=["hard-negative", "near-miss", mutation["desc"]],
        )
        results.append(ex)

    # ── Section-confusion negatives ──────────────────────────────────────────
    section_confusion_budget = count // 3
    for _ in range(section_confusion_budget):
        seed = rng.choice(nli_seeds)
        confusion = rng.choice(SECTION_CONFUSION_PAIRS)

        # The premise is about the correct_clause but we pair it with the
        # wrong_clause's hypothesis → contradiction or neutral
        if seed["clause"] != confusion["correct_clause"]:
            continue

        ex = SyntheticExample(
            clause=confusion["wrong_clause"],
            premise=seed["premise"],
            hypothesis=confusion["hypothesis_template"],
            label="contradiction" if seed["label"] == "entailment" else "neutral",
            scenario=seed.get("scenario", "") + "-section-confusion",
            difficulty=3,
            clause_ids=POPIA_CLAUSES.get(confusion["wrong_clause"], {}).get("sections", []),
            scenario_type="hard-negative",
            generation_method="syneg-section-confusion",
            source_id=f"{seed.get('_source_file', 'unknown')}:{seed.get('_source_idx', -1)}",
            tags=["hard-negative", "section-confusion", confusion["wrong_clause"]],
        )
        results.append(ex)

    # ── Partial-compliance negatives ─────────────────────────────────────────
    partial_budget = count // 3
    for _ in range(partial_budget):
        seed = rng.choice(entailment_seeds)
        suffix = rng.choice(PARTIAL_COMPLIANCE_SUFFIXES)
        new_premise = seed["premise"].rstrip(".") + "." + suffix
        clause_info = POPIA_CLAUSES.get(seed["clause"], {})

        ex = SyntheticExample(
            clause=seed["clause"],
            premise=new_premise,
            hypothesis=seed["hypothesis"],
            label="contradiction",
            scenario=seed.get("scenario", "") + "-partial",
            difficulty=4,
            clause_ids=clause_info.get("sections", []),
            scenario_type="hard-negative",
            generation_method="syneg-partial-compliance",
            source_id=f"{seed.get('_source_file', 'unknown')}:{seed.get('_source_idx', -1)}",
            tags=["hard-negative", "partial-compliance"],
        )
        results.append(ex)

    return results


# ── 3. Scenario diversification ─────────────────────────────────────────────

SCENARIO_PREMISE_TEMPLATES: dict[str, list[str]] = {
    "POPIA consent": [
        "{company_desc} is {action_verb} {data_type_desc} without obtaining explicit, informed consent from the data subjects.",
        "{company_desc} obtains consent for {action_verb} {data_type_desc} via a clear, unbundled opt-in form that explains the specific purpose.",
        "{company_desc} is {action_verb} {data_type_desc} and relies on a pre-ticked consent box buried in the terms and conditions.",
    ],
    "POPIA minimality / purpose limitation": [
        "{company_desc} collects {data_type_desc} along with extensive additional personal details that are not required for the stated purpose.",
        "{company_desc} collects only the {data_type_desc} strictly necessary for the declared purpose and retains it only for the stated period.",
        "{company_desc} originally collected {data_type_desc} for purpose A but is now {action_verb} it for an unrelated purpose B without updating the privacy notice.",
    ],
    "POPIA security safeguards": [
        "{company_desc} stores {data_type_desc} in an unencrypted database accessible to all staff via a shared password.",
        "{company_desc} protects {data_type_desc} with AES-256 encryption, role-based access control, MFA, and quarterly penetration testing.",
        "{company_desc} outsources the {action_verb} of {data_type_desc} to a third party without verifying the third party's security measures.",
    ],
    "POPIA breach notification": [
        "{company_desc} discovered that {data_type_desc} was accessed by an unauthorised person but delayed notifying the Regulator and affected data subjects for over six months.",
        "{company_desc} notified the Information Regulator and all affected data subjects in writing within 72 hours of confirming the breach of {data_type_desc}.",
        "A breach involving {data_type_desc} at {company_desc} is under investigation but has not yet been confirmed.",
    ],
    "POPIA cross-border transfers": [
        "{company_desc} transfers {data_type_desc} to a processor in a foreign jurisdiction without any binding data-protection agreement.",
        "{company_desc} transfers {data_type_desc} to a jurisdiction with adequate data-protection legislation under binding contractual clauses.",
        "{company_desc} hosts {data_type_desc} exclusively on South African servers with no replication abroad.",
    ],
    "POPIA general processing": [
        "{company_desc} processes {data_type_desc} without a documented lawful basis and without informing data subjects.",
        "{company_desc} has completed a privacy impact assessment, appointed an Information Officer, and maintains a processing register for {data_type_desc}.",
        "{company_desc} is {action_verb} {data_type_desc} in a manner that is disproportionate to the purpose.",
    ],
    "POPIA data subject rights": [
        "{company_desc} refuses all subject-access requests relating to {data_type_desc} unless accompanied by a court order.",
        "{company_desc} responds to every subject-access request within 21 days, provides the data free of charge, and notifies third parties of any corrections.",
        "{company_desc} has no documented process for handling requests from data subjects regarding {data_type_desc}.",
    ],
    "POPIA special personal information": [
        "{company_desc} processes {data_type_desc} — which qualifies as special personal information — without any of the §27 authorisation grounds.",
        "{company_desc} processes {data_type_desc} with explicit written consent and appropriate safeguards as required under §27.",
        "{company_desc} incidentally encounters {data_type_desc} during routine operations but does not specifically process it for any purpose.",
    ],
    "POPIA children's information": [
        "{company_desc} collects {data_type_desc} from users under 18 without obtaining parental or guardian consent.",
        "{company_desc} requires verified parental consent before processing any {data_type_desc} belonging to minors.",
        "{company_desc} provides services to families and may incidentally handle {data_type_desc} of minors.",
    ],
    "POPIA automated decision-making": [
        "{company_desc} uses automated profiling based on {data_type_desc} to make decisions with legal effect, without offering human review.",
        "{company_desc} uses automated scoring of {data_type_desc} but allows affected data subjects to request human reconsideration.",
        "{company_desc} uses an algorithm that processes {data_type_desc}, but it only generates recommendations for a human decision-maker.",
    ],
}

HYPOTHESIS_TEMPLATES_BY_CLAUSE: dict[str, list[tuple[str, str]]] = {
    "POPIA consent": [
        ("The responsible party has obtained valid, freely-given consent for the processing.", "entailment"),
        ("The responsible party has obtained valid, freely-given consent for the processing.", "contradiction"),
        ("The data subject has given informed consent.", "neutral"),
    ],
    "POPIA minimality / purpose limitation": [
        ("The data collected is adequate, relevant and not excessive for the stated purpose.", "entailment"),
        ("The data collected is adequate, relevant and not excessive for the stated purpose.", "contradiction"),
        ("The retention period is proportionate to the purpose.", "neutral"),
    ],
    "POPIA security safeguards": [
        ("The responsible party has appropriate technical and organisational safeguards.", "entailment"),
        ("The responsible party has appropriate technical and organisational safeguards.", "contradiction"),
        ("Security measures are in line with industry best practice.", "neutral"),
    ],
    "POPIA breach notification": [
        ("The responsible party notified the Regulator and data subjects as soon as reasonably possible.", "entailment"),
        ("The responsible party notified the Regulator and data subjects as soon as reasonably possible.", "contradiction"),
        ("A breach has occurred that triggers §22 obligations.", "neutral"),
    ],
    "POPIA cross-border transfers": [
        ("The cross-border transfer complies with at least one §72 ground.", "entailment"),
        ("The cross-border transfer complies with at least one §72 ground.", "contradiction"),
        ("Personal information is transferred outside South Africa.", "neutral"),
    ],
    "POPIA general processing": [
        ("The processing is lawful and reasonable under §9.", "entailment"),
        ("The processing is lawful and reasonable under §9.", "contradiction"),
        ("The responsible party has taken accountability steps.", "neutral"),
    ],
    "POPIA data subject rights": [
        ("The responsible party is honouring the data subject's rights under §23–§25.", "entailment"),
        ("The responsible party is honouring the data subject's rights under §23–§25.", "contradiction"),
        ("Data subjects can exercise their rights under POPIA.", "neutral"),
    ],
    "POPIA special personal information": [
        ("The responsible party has a lawful basis under §27 to process special personal information.", "entailment"),
        ("The responsible party has a lawful basis under §27 to process special personal information.", "contradiction"),
        ("Special personal information is involved in this processing.", "neutral"),
    ],
    "POPIA children's information": [
        ("The responsible party has obtained parental consent before processing the child's data.", "entailment"),
        ("The responsible party has obtained parental consent before processing the child's data.", "contradiction"),
        ("Personal information of minors may be involved.", "neutral"),
    ],
    "POPIA automated decision-making": [
        ("The data subject can request human review of the automated decision.", "entailment"),
        ("The data subject can request human review of the automated decision.", "contradiction"),
        ("Automated processing is being used.", "neutral"),
    ],
}


def diversify_scenarios(rng: random.Random, count: int = 200) -> list[SyntheticExample]:
    """Generate NLI pairs across industry × size × data-type × action combinations."""
    results: list[SyntheticExample] = []

    combos = list(itertools.product(INDUSTRIES, COMPANY_SIZES, DATA_TYPES, ACTIONS))
    rng.shuffle(combos)

    for combo in combos[:count]:
        (industry_name, industry_desc), (size_name, size_desc), \
            (dtype_name, dtype_desc), (action_name, action_verb) = combo

        clause = rng.choice(list(POPIA_CLAUSES.keys()))
        clause_info = POPIA_CLAUSES[clause]
        premise_templates = SCENARIO_PREMISE_TEMPLATES.get(clause, [])
        hyp_options = HYPOTHESIS_TEMPLATES_BY_CLAUSE.get(clause, [])

        if not premise_templates or not hyp_options:
            continue

        # Pick a premise template — index 0 is typically non-compliant,
        # index 1 is compliant, index 2 is ambiguous
        template_idx = rng.randint(0, len(premise_templates) - 1)
        premise_template = premise_templates[template_idx]

        company_desc = f"{industry_desc} ({size_desc})"
        premise = premise_template.format(
            company_desc=company_desc,
            action_verb=action_verb,
            data_type_desc=dtype_desc,
        )

        # Align hypothesis and label with the premise intent
        if template_idx == 0:  # non-compliant scenario
            hyp, _ = hyp_options[1]  # contradiction hypothesis
            label = "contradiction"
            difficulty = 2
        elif template_idx == 1:  # compliant scenario
            hyp, _ = hyp_options[0]  # entailment hypothesis
            label = "entailment"
            difficulty = 1
        else:  # ambiguous
            hyp, _ = hyp_options[2]  # neutral hypothesis
            label = "neutral"
            difficulty = 2

        scenario_tag = f"{industry_name}-{size_name}-{dtype_name}-{action_name}"

        ex = SyntheticExample(
            clause=clause,
            premise=premise,
            hypothesis=hyp,
            label=label,
            scenario=scenario_tag,
            difficulty=difficulty,
            clause_ids=clause_info.get("sections", []),
            scenario_type="diversified",
            generation_method="scenario-diversification",
            tags=[industry_name, size_name, dtype_name, action_name],
        )
        results.append(ex)

    return results


# ── 4. Curriculum-ordered instruct examples ──────────────────────────────────

CURRICULUM_TEMPLATES: dict[int, list[dict[str, str]]] = {
    1: [
        {
            "user": "What is the title of POPIA {section}?",
            "assistant": "POPIA {section} — {section_title}.",
            "tag": "section-lookup",
        },
        {
            "user": "Which POPIA condition does {clause} fall under?",
            "assistant": "{clause} falls under {sections_str}.",
            "tag": "clause-identification",
        },
    ],
    2: [
        {
            "user": "A {company_desc} is {action_verb} {data_type_desc}. Which POPIA sections apply?",
            "assistant": (
                "This scenario engages {clause} ({sections_str}). The responsible party must "
                "ensure that the {action_verb} of {data_type_desc} satisfies the conditions "
                "set out in these provisions."
            ),
            "tag": "multi-clause-identification",
        },
        {
            "user": (
                "Review this scenario against {clause}: {premise}"
            ),
            "assistant": (
                "This scenario {compliance_word} {sections_str}. The processing described "
                "— '{premise_short}' — {compliance_reason}."
            ),
            "tag": "scenario-review",
        },
    ],
    3: [
        {
            "user": (
                "A {company_desc} argues that POPIA {section} does not apply to their "
                "{action_verb} of {data_type_desc} because {exemption_claim}. "
                "Is this exemption valid?"
            ),
            "assistant": (
                "The claimed exemption under {section} would {exemption_result} in this case. "
                "{exemption_reasoning}"
            ),
            "tag": "exception-analysis",
        },
    ],
    4: [
        {
            "user": (
                "A {company_desc} is {action_verb} {data_type_desc} of minors and also "
                "transferring it to {foreign_country}. Compare the requirements under "
                "{clause} and POPIA §72."
            ),
            "assistant": (
                "This scenario engages two distinct POPIA regimes simultaneously:\n\n"
                "1. {clause} ({sections_str}): {clause_requirement}\n"
                "2. Cross-border transfers (§72): The transfer to {foreign_country} must "
                "satisfy at least one of the §72 grounds.\n\n"
                "Both sets of requirements must be met for the processing to be lawful."
            ),
            "tag": "cross-regulation",
        },
    ],
    5: [
        {
            "user": (
                "{edge_case_scenario} Analyse whether POPIA applies and, if so, "
                "which provisions are engaged."
            ),
            "assistant": (
                "This is an adversarial edge case. The analysis turns on several contested "
                "points:\n\n{edge_case_analysis}\n\n"
                "The Information Regulator has not issued definitive guidance on this exact "
                "fact pattern, so the answer involves interpretation."
            ),
            "tag": "adversarial",
        },
    ],
}

SECTION_TITLES: dict[str, str] = {
    "§5": "Rights of data subjects",
    "§8": "Application",
    "§9": "Lawfulness of processing",
    "§10": "Minimality",
    "§11": "Consent, justification and objection",
    "§12": "Collection directly from data subject",
    "§13": "Collection for specific purpose",
    "§14": "Retention and restriction of records",
    "§15": "Further processing",
    "§18": "Notification to data subject",
    "§19": "Security measures on condition of processing",
    "§20": "Information processed by operator",
    "§21": "Security compromises",
    "§22": "Notification of security compromises",
    "§23": "Access to personal information",
    "§24": "Correction of personal information",
    "§25": "Manner of access",
    "§26": "Prohibition on processing special personal information",
    "§27": "Authorisation concerning special personal information",
    "§28": "Authorisation concerning the processing of special personal information for historical, statistical or research purposes",
    "§29": "Criminal behaviour, biometric information and children",
    "§30": "Processing of personal information of children",
    "§31": "Authorisations concerning the processing of personal information of children",
    "§32": "Processing by establishment in the Republic",
    "§34": "Prohibition of processing concerning children",
    "§35": "Competent person",
    "§69": "Direct marketing by means of unsolicited electronic communications",
    "§71": "Automated decision-making",
    "§72": "Transfer of personal information outside Republic",
    "§101": "Breach of confidentiality",
}

EXEMPTION_CLAIMS = [
    ("the data is publicly available", "not succeed", "Public availability does not negate POPIA obligations — §6 still requires lawful processing of publicly available personal information."),
    ("they have a legitimate interest", "require careful balancing", "Legitimate interest under §11(1)(f) requires a proportionality assessment between the responsible party's interest and the data subject's right to privacy."),
    ("the processing is for journalistic purposes", "only succeed if genuine journalism is involved", "The §7 exemption is narrow and applies only to processing solely for journalistic, literary or artistic expression."),
    ("consent was obtained before POPIA commenced", "need to be reassessed", "Pre-POPIA consent may not meet the standard of 'specific, voluntary and informed' consent required under §11."),
]

EDGE_CASE_SCENARIOS = [
    "A South African company's employee uses a personal device to access customer data while on holiday abroad.",
    "An AI model trained on South African personal information is deployed in a jurisdiction without data protection laws.",
    "A data subject requests deletion, but the responsible party argues the data is needed for a pending CCMA dispute.",
    "A company captures biometric data (facial recognition) for access control and also uses it for productivity monitoring without separate consent.",
    "An insurer uses credit-score data from a bureau — which was originally collected for a different purpose — to set premiums.",
]

EDGE_CASE_ANALYSES = [
    "1. Whether the extraterritorial provisions in §3 apply.\n2. Whether a cross-border transfer occurred.\n3. Whether the employee's actions constitute processing by the responsible party.",
    "1. Whether the training constitutes 'processing' under POPIA.\n2. Whether §72 applies to model weights as opposed to raw data.\n3. Whether de-identification under §6(1) resolves the issue.",
    "1. Whether §14's retention provision permits keeping data for litigation.\n2. Whether §11's legitimate-interest ground covers pending disputes.\n3. The interplay between POPIA and labour legislation.",
    "1. Whether §27's biometric-data provisions cover both purposes.\n2. Whether the original consent is specific enough for productivity monitoring.\n3. Whether the employee has a right to object under §11(3).",
    "1. Whether §15 (further processing) permits the insurer's use.\n2. Whether the credit bureau's original consent covers this new purpose.\n3. Whether the insurer qualifies as a responsible party or operator.",
]


def generate_curriculum_instruct(
    nli_seeds: list[dict], rng: random.Random, count: int = 120
) -> list[SyntheticExample]:
    """Generate instruct examples tagged with curriculum difficulty levels 1-5."""
    results: list[SyntheticExample] = []
    # Distribute roughly evenly across levels, slightly more at levels 2-3
    level_weights = {1: 0.15, 2: 0.25, 3: 0.25, 4: 0.20, 5: 0.15}
    level_counts = {lvl: max(1, int(count * w)) for lvl, w in level_weights.items()}

    for level, n in level_counts.items():
        templates = CURRICULUM_TEMPLATES.get(level, [])
        if not templates:
            continue

        for _ in range(n):
            template = rng.choice(templates)
            clause = rng.choice(list(POPIA_CLAUSES.keys()))
            clause_info = POPIA_CLAUSES[clause]
            sections = clause_info["sections"]
            sections_str = ", ".join(sections)
            _, industry_desc = rng.choice(INDUSTRIES)
            _, company_desc = rng.choice(COMPANY_SIZES)
            _, dtype_desc = rng.choice(DATA_TYPES)
            _, action_verb = rng.choice(ACTIONS)
            section = rng.choice(sections)
            foreign_country = rng.choice(FOREIGN_COUNTRIES)

            seed = rng.choice(nli_seeds) if nli_seeds else {}
            premise = seed.get("premise", "A company processes personal information.")
            premise_short = premise[:100] + "..." if len(premise) > 100 else premise
            label = seed.get("label", "neutral")

            compliance_word = (
                "aligns with" if label == "entailment"
                else "does NOT comply with" if label == "contradiction"
                else "is not directly governed by"
            )
            compliance_reason = (
                "satisfies the requirement"
                if label == "entailment"
                else "violates the requirement"
                if label == "contradiction"
                else "does not raise the specific question this section addresses"
            )

            exemption_claim, exemption_result, exemption_reasoning = rng.choice(EXEMPTION_CLAIMS)
            edge_idx = rng.randint(0, len(EDGE_CASE_SCENARIOS) - 1)
            edge_case_scenario = EDGE_CASE_SCENARIOS[edge_idx]
            edge_case_analysis = EDGE_CASE_ANALYSES[edge_idx]

            section_title = SECTION_TITLES.get(section, "Unknown")

            clause_requirement = (
                f"requires that the processing of this data type satisfies the conditions "
                f"under {sections_str}"
            )

            fmt_kwargs = dict(
                clause=clause,
                sections_str=sections_str,
                section=section,
                section_title=section_title,
                company_desc=f"{industry_desc} ({company_desc})",
                industry_desc=industry_desc,
                action_verb=action_verb,
                data_type_desc=dtype_desc,
                premise=premise,
                premise_short=premise_short,
                compliance_word=compliance_word,
                compliance_reason=compliance_reason,
                foreign_country=foreign_country,
                exemption_claim=exemption_claim,
                exemption_result=exemption_result,
                exemption_reasoning=exemption_reasoning,
                edge_case_scenario=edge_case_scenario,
                edge_case_analysis=edge_case_analysis,
                clause_requirement=clause_requirement,
            )

            try:
                user_msg = template["user"].format(**fmt_kwargs)
                asst_msg = template["assistant"].format(**fmt_kwargs)
            except KeyError:
                continue

            ex = SyntheticExample(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": asst_msg},
                ],
                difficulty=level,
                clause_ids=sections,
                scenario_type="curriculum",
                generation_method=f"curriculum-L{level}-{template['tag']}",
                tags=[f"level-{level}", template["tag"]],
            )
            results.append(ex)

    return results


# ── Quality filter ───────────────────────────────────────────────────────────

MIN_PREMISE_LEN = 30
MAX_PREMISE_LEN = 800
MIN_HYP_LEN = 15
MAX_HYP_LEN = 300
MIN_MSG_LEN = 20
MAX_MSG_LEN = 2000


def content_hash(ex: SyntheticExample) -> str:
    """Produce a dedup hash based on meaningful content."""
    if ex.messages:
        key = "|".join(m.get("content", "") for m in ex.messages)
    else:
        key = f"{ex.premise}|{ex.hypothesis}|{ex.label}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def quality_filter(examples: list[SyntheticExample]) -> list[SyntheticExample]:
    """Apply dedup, length bounds, and format-validity checks."""
    seen_hashes: set[str] = set()
    passed: list[SyntheticExample] = []
    stats: Counter = Counter()

    for ex in examples:
        ch = content_hash(ex)

        # Duplicate check
        if ch in seen_hashes:
            stats["duplicate"] += 1
            continue
        seen_hashes.add(ch)

        # NLI format checks
        if ex.messages is None:
            if not ex.premise or not ex.hypothesis or not ex.label:
                stats["missing_nli_fields"] += 1
                continue
            if not (MIN_PREMISE_LEN <= len(ex.premise) <= MAX_PREMISE_LEN):
                stats["premise_length"] += 1
                continue
            if not (MIN_HYP_LEN <= len(ex.hypothesis) <= MAX_HYP_LEN):
                stats["hypothesis_length"] += 1
                continue
            if ex.label not in ("entailment", "contradiction", "neutral"):
                stats["invalid_label"] += 1
                continue

        # Instruct format checks
        if ex.messages is not None:
            if len(ex.messages) < 2:
                stats["too_few_messages"] += 1
                continue
            for msg in ex.messages:
                if "role" not in msg or "content" not in msg:
                    stats["malformed_message"] += 1
                    break
                if msg["role"] in ("user", "assistant"):
                    if not (MIN_MSG_LEN <= len(msg["content"]) <= MAX_MSG_LEN):
                        stats["message_length"] += 1
                        break
            else:
                passed.append(ex)
                continue
            continue  # A message check failed

        passed.append(ex)

    log.info("Quality filter: %d → %d examples", len(examples), len(passed))
    for reason, count in stats.most_common():
        log.info("  Filtered out %d for: %s", count, reason)
    return passed


# ── Stats reporting ──────────────────────────────────────────────────────────

def print_stats(examples: list[SyntheticExample]) -> None:
    """Print generation statistics."""
    total = len(examples)
    by_method = Counter(ex.generation_method for ex in examples)
    by_difficulty = Counter(ex.difficulty for ex in examples)
    by_clause = Counter(cid for ex in examples for cid in ex.clause_ids)
    by_scenario_type = Counter(ex.scenario_type for ex in examples)
    by_label = Counter(ex.label for ex in examples if ex.label)
    by_format = Counter("instruct" if ex.messages else "nli" for ex in examples)

    print("\n" + "=" * 72)
    print(f"  SYNTHETIC DATA GENERATION REPORT")
    print(f"  Total examples: {total}")
    print("=" * 72)

    print(f"\n{'─'*36}")
    print("  BY FORMAT")
    print(f"{'─'*36}")
    for fmt, cnt in by_format.most_common():
        print(f"    {fmt:<20s}  {cnt:>5d}  ({cnt/total*100:5.1f}%)")

    print(f"\n{'─'*36}")
    print("  BY GENERATION METHOD")
    print(f"{'─'*36}")
    for method, cnt in by_method.most_common():
        print(f"    {method:<40s}  {cnt:>5d}  ({cnt/total*100:5.1f}%)")

    print(f"\n{'─'*36}")
    print("  BY DIFFICULTY LEVEL")
    print(f"{'─'*36}")
    for level in sorted(by_difficulty):
        cnt = by_difficulty[level]
        bar = "█" * int(cnt / total * 40)
        print(f"    Level {level}  {cnt:>5d}  ({cnt/total*100:5.1f}%)  {bar}")

    print(f"\n{'─'*36}")
    print("  BY POPIA SECTION")
    print(f"{'─'*36}")
    for clause, cnt in by_clause.most_common():
        print(f"    {clause:<8s}  {cnt:>5d}")

    print(f"\n{'─'*36}")
    print("  BY NLI LABEL (NLI examples only)")
    print(f"{'─'*36}")
    for label, cnt in by_label.most_common():
        print(f"    {label:<15s}  {cnt:>5d}")

    print(f"\n{'─'*36}")
    print("  BY SCENARIO TYPE")
    print(f"{'─'*36}")
    for stype, cnt in by_scenario_type.most_common():
        print(f"    {stype:<20s}  {cnt:>5d}")

    print("\n" + "=" * 72)


# ── Main pipeline ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic POPIA training data.")
    parser.add_argument(
        "--seed-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing seed JSONL files (default: data/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/popia_synthetic_v1.jsonl"),
        help="Output JSONL path (default: data/popia_synthetic_v1.jsonl)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--evol-count",
        type=int,
        default=80,
        help="Number of Evol-Instruct examples to generate (default: 80)",
    )
    parser.add_argument(
        "--neg-count",
        type=int,
        default=100,
        help="Number of hard-negative examples to generate (default: 100)",
    )
    parser.add_argument(
        "--diverse-count",
        type=int,
        default=200,
        help="Number of diversified scenarios to generate (default: 200)",
    )
    parser.add_argument(
        "--curriculum-count",
        type=int,
        default=120,
        help="Number of curriculum instruct examples to generate (default: 120)",
    )
    args = parser.parse_args()
    rng = random.Random(args.random_seed)

    # ── Load seeds ───────────────────────────────────────────────────────────
    log.info("Loading seed data from %s", args.seed_dir)
    nli_seeds, instruct_seeds = load_seeds(args.seed_dir)
    log.info("Seeds loaded: %d NLI, %d instruct", len(nli_seeds), len(instruct_seeds))

    if not nli_seeds:
        log.error("No NLI seeds found in %s — cannot proceed.", args.seed_dir)
        sys.exit(1)

    # ── Generate ─────────────────────────────────────────────────────────────
    all_examples: list[SyntheticExample] = []

    log.info("Strategy 1: Evol-Instruct complexity evolution (%d target)", args.evol_count)
    evol = evolve_with_constraints(nli_seeds, rng, count=args.evol_count)
    all_examples.extend(evol)
    log.info("  → produced %d examples", len(evol))

    log.info("Strategy 2: Hard-negative generation (%d target)", args.neg_count)
    negs = generate_hard_negatives(nli_seeds, rng, count=args.neg_count)
    all_examples.extend(negs)
    log.info("  → produced %d examples", len(negs))

    log.info("Strategy 3: Scenario diversification (%d target)", args.diverse_count)
    diverse = diversify_scenarios(rng, count=args.diverse_count)
    all_examples.extend(diverse)
    log.info("  → produced %d examples", len(diverse))

    log.info("Strategy 4: Curriculum-ordered instruct (%d target)", args.curriculum_count)
    curric = generate_curriculum_instruct(nli_seeds, rng, count=args.curriculum_count)
    all_examples.extend(curric)
    log.info("  → produced %d examples", len(curric))

    # ── Quality filter ───────────────────────────────────────────────────────
    log.info("Running quality filters …")
    filtered = quality_filter(all_examples)

    # ── Sort by difficulty for curriculum ordering ────────────────────────────
    filtered.sort(key=lambda ex: (ex.difficulty, ex.generation_method))

    # ── Write output ─────────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for ex in filtered:
            f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
    log.info("Wrote %d examples to %s", len(filtered), args.output)

    # ── Print stats ──────────────────────────────────────────────────────────
    print_stats(filtered)


if __name__ == "__main__":
    main()
