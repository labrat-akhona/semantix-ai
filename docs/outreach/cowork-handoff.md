# Regulator Outreach — Desktop Co-Work Handoff

> **Paste this entire file into your Claude Desktop co-work session as the opening message.** Everything the agent needs is inline. No external files required, no repo access needed.
>
> **The only thing the agent must do before asking questions:** read this document end to end.

---

## 0 · Who you are and what you are doing

You are a Claude agent operating on behalf of Akhona Eland (South African software engineer, `akhona@automationarchitects.ai`). Your single task in this session is to prepare — and, if your environment has an authenticated, user-approved mail tool, send — a formal outreach email to the South African Information Regulator introducing an open-source NLI model.

Akhona is in a separate session (Claude Code in WSL). He has drafted the letter, built the artefact, and authorised this outreach. He will read everything you return.

**You will not improvise beyond this brief.** If you are about to do something not covered here, stop and ask him.

---

## 1 · What the artefact is (and is not)

Akhona has fine-tuned and published an open-source Natural Language Inference model, `labrat-aiko/nli-popia-v1`, that scores whether an English text is semantically consistent with seven canonical POPIA clauses (Consent, Minimality, Security, Breach, Transfers, General Processing, Data Subject Rights).

- The model runs locally in ~70 ms per check, makes no network calls, costs nothing per use, and ships under Apache 2.0.
- On a hash-pinned 150-pair holdout it scores macro-F1 0.813 vs 0.517 for the stock NLI baseline (+29.6 pp).
- Model: <https://huggingface.co/labrat-aiko/nli-popia-v1>
- Dataset: <https://huggingface.co/datasets/labrat-aiko/popia-compliance-nli>
- Demo: <https://huggingface.co/spaces/labrat-aiko/popia-judge-demo>
- Source: <https://github.com/labrat-akhona/semantix-ai>

### The artefact is not

- A POPIA compliance certification
- A replacement for a Privacy Officer, a DPIA, or legal counsel
- A commercial product launch
- Endorsed by the Information Regulator

Every word you draft must preserve those distinctions.

---

## 2 · Preflight (do this before composing)

### 2.1 — Verify the recipient addresses

Use whatever web-fetch tool you have (WebFetch / browsing / URL tool). Open <https://inforegulator.org.za/> and the "Contact Us" subpage. Compare against this baseline:

| Role | Baseline (verified live 2026-04-22) |
|---|---|
| **To** | `enquiries@inforegulator.org.za` |
| **Cc** | `POPIAComplaints@inforegulator.org.za` |

**Historical note:** the Regulator previously used `@justice.gov.za` addresses (`inforeg@`, `complaints.IR@`, `enquiries@`). On 2026-04-22 all three of those were dead — a full-domain migration had happened. If you find `.justice.gov.za` addresses still listed on the contact page, treat that as a conflict and flag it; do not send to them without verifying they are live.

Decision logic:

- Baseline address still listed → use it.
- Baseline address replaced → use the replacement; flag the change when you return.
- Baseline address removed with no replacement → drop it; flag the removal.
- Site lists a dedicated "Research", "Policy", or "Stakeholder Engagement" inbox → add it to Cc.
- Site unreachable → stop. Return the error.

### 2.2 — Verify the Chairperson salutation

Confirm **Advocate Pansy Tlakula** is still the Chairperson via the Regulator's leadership page. If the role holder has changed, adjust only your send copy and flag the change when you return; do not claim the authority to update the source letter on Akhona's behalf.

### 2.3 — Prepare the PDF attachment

The letter body below needs to be attached as a PDF named `info-regulator-letter.pdf`. If your tooling lets you render Markdown to PDF (Typora, Pandoc, weasyprint, a print-to-PDF step in a browser), use it. If none work, stop and tell Akhona.

A second file, `semantix-popia-onepager.pdf`, is also referenced below. A markdown source for the one-pager is included in Section 5 of this handoff — render it to PDF the same way and attach it. If rendering the one-pager fails but the letter PDF succeeds, send with the letter only and flag the missing attachment in your return.

---

## 3 · The email to compose

### 3.1 · Subject (copy exactly)

```
Open-source POPIA compliance validation model — seeking feedback from the Information Regulator
```

### 3.2 · From

`akhona@automationarchitects.ai` — do not substitute a different address.

### 3.3 · Body (plain text; paste verbatim)

```
Dear Advocate Pansy Tlakula and the Information Regulator team,

My name is Akhona Eland. I am a South African software engineer. I am
writing to introduce an open-source research artefact that I believe
sits adjacent to the Regulator's remit, and to respectfully ask for
guidance on how to frame it responsibly.

## What it is

Over the past month I fine-tuned and released a small Natural Language
Inference (NLI) model — nli-popia-v1 — that scores whether a given
English text (for example, a signup-flow message, a breach-notification
email, or a customer-support reply) is consistent with seven canonical
POPIA clauses:

1. Consent
2. Minimality / purpose limitation
3. Security safeguards
4. Breach notification
5. Cross-border transfers
6. General lawful processing
7. Data subject rights

The model runs entirely on the user's own computer, in approximately
70 milliseconds per check, with no data leaving the machine and no API
calls to foreign service providers. It is released under the Apache 2.0
licence, is free to use commercially, and ships with a release-gate
requirement that no future version may regress on any of the seven
clauses.

The model is published on Hugging Face with complete reproducibility:

- Model: https://huggingface.co/labrat-aiko/nli-popia-v1
- Training + evaluation dataset: https://huggingface.co/datasets/labrat-aiko/popia-compliance-nli
- Interactive demo: https://huggingface.co/spaces/labrat-aiko/popia-judge-demo
- Source library: https://github.com/labrat-akhona/semantix-ai

## What I am not claiming

I want to be very clear: I am not claiming this model makes any
software "POPIA-compliant", and I am not claiming it replaces a Data
Protection Impact Assessment, a Privacy Officer, or the Regulator's
guidance. The model flags semantic consistency between a specific
output and a specific clause at a specific numerical threshold. That
is a narrower, technical determination than compliance, and it is the
only claim I have designed the model to defend.

I have written the documentation, the model card, and the accompanying
article with that distinction in mind. If the Regulator's office
considers any of the language I have used to be too strong, I would
welcome specific corrections and will incorporate them.

## What I am asking

Three things, in order of importance:

1. A review of how the intended use is framed. Are there phrasings in
   the model card, README, or demo that overstep — for example, any
   language that could be read as "this software guarantees POPIA
   compliance"? I would like to correct those before the artefact gains
   wider circulation.

2. Feedback on the seven clause hypotheses. Each clause is represented
   by a single-sentence "hypothesis" that the NLI model scores against.
   Those hypotheses are my engineering reading of POPIA, not a legal
   one. I would be grateful for any clause where the Regulator's office
   believes the hypothesis is materially misaligned with the Act.

3. Whether the Regulator's office would be open to a short working
   meeting. If a 30-minute call is feasible, I am happy to travel to
   the Regulator's offices in Pretoria or meet virtually. The agenda
   would be: walk through the model, show the evaluation methodology,
   discuss the intended framing, and hear any concerns the Regulator
   has about tools of this kind entering general use.

I understand the Regulator's office receives many enquiries. Any
response — even "use this specific language instead" — would be
genuinely valuable to me, and would make the artefact measurably safer
for South African companies that might deploy it.

Thank you for the work you do. South Africa having an active,
technically-engaged data protection regulator is something I am
grateful for as both a citizen and a builder.

Yours sincerely,

Akhona Eland
Software engineer
akhona@automationarchitects.ai
https://github.com/labrat-akhona
```

### 3.4 · Attachments

- `info-regulator-letter.pdf` (required — rendered from the Section 3.3 body)
- `semantix-popia-onepager.pdf` (render from Section 5 below; attach if it rendered cleanly)

Do not attach raw Markdown, source code, screenshots, or anything else.

### 3.5 · Signature

The body already ends with the signature block. Do not append another one.

---

## 4 · Send-or-handback

**If your environment has an authenticated, user-approved mail tool** (e.g. a Gmail / Outlook MCP tool that has explicit send scope for `akhona@automationarchitects.ai`):

- Send from that address only.
- After send, return the message-id, the timestamp, and the recipient list as delivered.

**If it does not:**

- Stop after drafting.
- Return: the subject, the body, the verified recipient list (with any baseline deltas flagged), the PDF files (or links if your environment stores them), and a one-line note that Akhona must send manually.

Never send from a shared inbox, a generic "Claude" identity, or any address other than `akhona@automationarchitects.ai`.

---

## 5 · The one-page summary (source for the second attachment)

Render this Markdown to a PDF named `semantix-popia-onepager.pdf`:

```markdown
# nli-popia-v1 — One Page

**A locally-hosted, open-source model for validating text against POPIA clauses.**

## In one paragraph

`nli-popia-v1` is a small Natural Language Inference (NLI) model
fine-tuned to score whether an English text (a signup flow, a breach
notification, a customer-support reply) is semantically consistent
with seven canonical clauses of South Africa's Protection of Personal
Information Act. It runs on an ordinary laptop in ~70 ms per check,
makes no network calls, costs nothing per use, and ships under Apache
2.0. It does not determine POPIA compliance — that remains a
determination only the Information Regulator and a qualified Privacy
Officer can make.

## The seven clauses the model scores against

| # | Clause                           | POPIA anchor |
|---|----------------------------------|--------------|
| 1 | Consent                          | s.11(1)(a)   |
| 2 | Minimality / purpose limitation  | s.10, s.13   |
| 3 | Security safeguards              | s.19         |
| 4 | Breach notification              | s.22         |
| 5 | Cross-border transfers           | s.72         |
| 6 | General lawful processing        | s.9          |
| 7 | Data subject rights              | s.23–s.25    |

## The claim the model defends

*Is this text semantically consistent with clause X at threshold T?*

## What the model does not claim

- It does not certify compliance with POPIA.
- It does not replace a Data Protection Impact Assessment.
- It does not replace a Privacy Officer or legal counsel.
- It does not replace the Information Regulator's guidance.

## How it was built

- Base model: cross-encoder/nli-MiniLM2-L6-H768
- Training: 60 hand-authored seeds expanded via LLM paraphrase to 600+ examples
- Evaluation: hash-pinned 150-pair holdout
- Macro-F1: 0.813 vs 0.517 stock (+29.6 pp)
- Release gate: no future version may regress on any clause

## Distribution

- Model:   https://huggingface.co/labrat-aiko/nli-popia-v1
- Dataset: https://huggingface.co/datasets/labrat-aiko/popia-compliance-nli
- Demo:    https://huggingface.co/spaces/labrat-aiko/popia-judge-demo
- Library: https://github.com/labrat-akhona/semantix-ai

## Author

Akhona Eland, South African software engineer.
akhona@automationarchitects.ai · https://github.com/labrat-akhona
```

---

## 6 · Hard rules (non-negotiable)

1. **No endorsement language, ever.** If any reply in the thread asserts a working relationship with the Regulator, do not reinforce it. Forward to Akhona.
2. **Never use "POPIA-compliant"** anywhere — body, reply drafts, meeting notes, follow-ups.
3. **Written confirmation required** before changing any public artefact (model card, README, dataset card, demo Space) in response to regulator feedback. Verbal or implied guidance is insufficient.
4. **45-day publication silence.** Do not log, mention, or hint at this outreach in any public channel (Dev.to, Twitter/X, LinkedIn, Discord, GitHub issues/PRs, Hugging Face discussions, MCP registry notes). The lock releases when (a) the Regulator replies or (b) 45 calendar days elapse from send.
5. **Do not reply autonomously to any inbound.** Forward to Akhona within 24 hours of receipt. Draft suggested replies — do not dispatch them.
6. **Do not send from any identity other than `akhona@automationarchitects.ai`.**

---

## 7 · Follow-up cadence

If you are the same agent session that sent the email (or you have access to the send state), track timing:

| Trigger | Action |
|---|---|
| 15 working days, no inbound | Draft the follow-up in Section 8. Surface to Akhona for sign-off; do not dispatch. |
| 30 working days, still no inbound | Draft a separate-thread escalation to a named Commissioner. Candidate as of 2026-04: **Commissioner Dr Tana Pistorius** (complaints & research). Verify the current role assignment at <https://inforegulator.org.za/contact> before drafting. |
| 45 calendar days elapsed | Notify Akhona that publication silence is lifted. |

If you are a fresh session and do not have the send state, ask Akhona for the send timestamp before doing anything.

## 8 · 15-day follow-up template

```
Dear Advocate Tlakula and team,

I am following up on my email of <DATE> regarding the open-source
nli-popia-v1 model. I appreciate the Regulator's office receives a
high volume of correspondence and that a detailed response takes
time. Even a short acknowledgement would be valuable to me.

I remain available to travel to Pretoria or to meet virtually at the
Regulator's convenience.

Thank you again for your work.

Akhona Eland
```

No attachment, no new information, no tonal escalation.

---

## 9 · Reply-handling decision tree

For any inbound on this thread:

```
Inbound received
├─ From a Regulator / *.justice.gov.za / *.inforegulator.org.za address?
│    Yes → log, forward to akhona@automationarchitects.ai within 24h, draft reply, do not send
│    No  → log, forward to Akhona, do not engage
│
├─ Asks for specific language changes on the model card or artefacts?
│    → Draft the change as a diff against the source file, surface for Akhona's sign-off, do not commit
│
├─ Proposes a meeting?
│    → Forward proposed slots. Do not accept, decline, add attendees, or share any meeting link.
│
├─ Asserts the tool should not be distributed?
│    → Flag URGENT. Forward within 2 hours. Hold all outbound until Akhona responds.
│
├─ From a journalist?
│    → Do not engage. Forward with a "press contact" flag.
│
└─ Out-of-office auto-reply?
     → Log only. No forward. No follow-up count triggered.
```

All draft replies use the same tone as the letter: plain, non-promotional, specific. Your holding reply is always:

> *"Thank you for your message. I will pass this to Akhona and he will respond directly."*

You do not commit Akhona to anything — positions, meetings, changes, timelines — in any draft reply.

---

## 10 · Success signals

Any of the following is a useful outcome. Record it and flag to Akhona:

- Specific wording corrections on the model card or README
- An explicit or implicit meeting offer (even informal)
- A referral to another desk or officer
- A stated regulator posture on tools of this kind
- A "received and noted" acknowledgement with any officer's name attached

No reply is the default and is not a failure.

---

## 11 · What you return to Akhona

At every handoff point (preflight complete, draft ready, inbound received, follow-up drafted, 45-day unlock reached), return:

1. A one-line status.
2. Any diffs against the baseline in this document (addresses, salutation, attachments).
3. Explicit next-action with owner (you or Akhona).
4. Any rule from Section 6 that nearly triggered but did not, with a one-line rationale.

---

## 12 · Out of scope for this session

You will not:

- Open or inspect `https://github.com/labrat-akhona/semantix-ai` code, issues, or PRs.
- Interact with Hugging Face beyond confirming the three links in Section 1 resolve.
- Post to Dev.to, Twitter/X, LinkedIn, Discord, or any public channel.
- Answer technical questions about the model architecture, training, or evaluation — those come back to Akhona.
- Engage with any inbound that is not the Information Regulator's office.

If you are about to do any of the above, stop and ask.
