# Community-list PRs — Round 2 candidates

Research output from a dispatched subagent on 2026-04-21. Eight real, actively-maintained awesome lists verified (all with commits within the last 60 days). Ranked by merge likelihood. Complements `community-list-prs.md` (round 1).

## Already submitted (round 1)

- ✅ `punkpeye/awesome-mcp-servers` → PR [#5200](https://github.com/punkpeye/awesome-mcp-servers/pull/5200)
- ✅ `Hannibal046/Awesome-LLM` → PR [#506](https://github.com/Hannibal046/Awesome-LLM/pull/506)
- ✅ `steven2358/awesome-generative-ai` → PR [#662](https://github.com/steven2358/awesome-generative-ai/pull/662)

## Round 2 candidates

| # | Repo | Stars | Last commit | Target section | Placement | Why it fits | Risk |
|---|------|-------|-------------|----------------|-----------|-------------|------|
| 1 | **kyrolabs/awesome-agents** | 2.2k | 2026-04-18 | `## Testing and Evaluation` | Append-at-end (thematic, not alphabetical) | Dedicated "Testing and Evaluation" bucket is exactly semantix's pitch: deterministic validation of agent outputs. | CONTRIBUTING.md; clear fit + maintained repo required. **Low risk.** |
| 2 | **tensorchord/Awesome-LLMOps** | 5.7k | 2026-04-06 | `### Observability` | Alphabetical | LLM-output validation with hash-chained audit trail maps to Observability + guardrails scope. | Curates heavily. **Moderate risk** (selective). |
| 3 | **AthenaCore/AwesomeResponsibleAI** | 121 | 2026-04-17 | `## Tools` (Interactive and Practical Tools) | Alphabetical | Compliance-first positioning (POPIA/GDPR/EU AI Act) + tamper-evident audit trail is textbook Responsible-AI. Sister entry "AIR Blackbox" is almost identical scope. | Single maintainer. **Low risk.** |
| 4 | **EthicalML/awesome-artificial-intelligence-regulation** | 1.4k | 2026-04-18 | `Interactive and Practical Tools` | Thematic grouping | EU AI Act alignment + audit trail match regulation-ecosystem theme. | No explicit contrib rules; maintainer (Alejandro Saucedo) is selective. **Moderate risk.** |
| 5 | ~~**ganarajpr/awesome-dspy**~~ | 544 | 2025-12-10 | `## Projects` | Append | semantix ships first-party DSPy integration. | ❌ **Stale (>60 days).** Skip or submit as issue. |
| 6 | **wearetyomsmnv/Awesome-LLMSecOps** | 97 | 2026-04-16 | `## 🛡️Defense` (Security-by-Design) | Hierarchical | Hash-chained JSON-LD audit + NLI-based validation is a defensive control for LLM pipelines. | Small but active. **Low risk.** |
| 7 | **Vvkmnn/awesome-ai-eval** | 71 | 2026-03-25 | `### Evaluators and Test Harnesses` | Alphabetical | Explicit scope match: evaluating AI quality. pytest-semantix is a near-perfect test-harness entry. | Very small/new but responsive. **Low risk.** |
| 8 | **kyrolabs/awesome-langchain** | 9.3k | 2026-04-03 | `### Tools > Services` | Alphabetical | LangChain integration ships in semantix. | Strict guidelines — "brand new repo with no history" auto-close. At v0.1.12 this is a **high risk**. Hold until more releases. |

## Rejected (research notes)

- `onejune2018/Awesome-LLM-Eval` — last commit 2025-11-24 (stale)
- `jihoo-kim/awesome-production-llm` — last commit 2024-12-31 (stale)
- `dsfsi/awesome-africanlp` — NLP research focus, not tooling
- `andausman/awesome-african-datasets` — datasets only

No strong awesome-list match for the SA beachhead angle — that motion is better served via Deep Learning Indaba / SACAIR / community channels, not lists.

## Submission order

**Phase 2a (low friction, submit soon):**
1. #1 — `kyrolabs/awesome-agents`
2. #3 — `AthenaCore/AwesomeResponsibleAI`
3. #7 — `Vvkmnn/awesome-ai-eval`
4. #6 — `wearetyomsmnv/Awesome-LLMSecOps`

**Phase 2b (hold until we have a blog post / case study to cite):**
5. #2 — `tensorchord/Awesome-LLMOps`
6. #4 — `EthicalML/awesome-artificial-intelligence-regulation` (stretch)

**Phase 2c (hold until v0.2.x + more traction):**
7. #8 — `kyrolabs/awesome-langchain`

**Skip:** #5 (stale).

## Cadence

Don't open all four of phase 2a on the same day — looks like campaigning. Space by 24-48 hours. Ideally, wait for one of round-1 PRs (awesome-mcp-servers #5200 is highest-probability) to get a review signal before round 2.
