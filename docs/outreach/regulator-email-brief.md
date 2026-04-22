# Regulator Email — Agent Brief

> **For a Claude agent session operating on Akhona's behalf.** This brief is self-contained — do not request prior context. Read it once, then execute in order. Do not email anything yourself unless the user's environment gives you an authenticated mail tool; otherwise draft and return for human send.

---

## Context you need

Akhona Eland (South African software engineer, `akhona@automationarchitects.ai`, <https://github.com/labrat-akhona>) has fine-tuned and published an open-source NLI model (`labrat-aiko/nli-popia-v1`) that scores whether English text is consistent with seven canonical POPIA clauses. He is writing to the SA Information Regulator to introduce the artefact.

**The email is explicitly not:** a request for endorsement, a compliance claim, a press release, a product launch, or a regulatory petition.

**The email is:** a courteous introduction with three asks — framing review, clause-hypothesis feedback, and a short working meeting.

If at any point your actions would move outside those four bullets, stop and ask the user.

---

## Files to read (in order)

1. `docs/outreach/info-regulator-letter.md` — the letter body. Use `Read`. This is authoritative; do not paraphrase it.
2. This brief — you are reading it now.

Do not read anything in `semantix/`, `data/`, `scripts/`, or `docs/superpowers/`. That is the engineering of the artefact. If a reply asks a technical question, stop and hand back to Akhona.

---

## Preflight (execute before composing)

### 1. Verify the recipient addresses

Use `WebFetch` on <https://inforegulator.org.za/> and the "Contact Us" subpage. Extract currently-listed email addresses and match against the baseline:

| Role | Baseline (verified live 2026-04-22) |
|---|---|
| **To** | `enquiries@inforegulator.org.za` |
| **Cc** | `POPIAComplaints@inforegulator.org.za` |

**Historical note:** the `@justice.gov.za` addresses listed in earlier drafts of this brief (`inforeg@`, `complaints.IR@`, `enquiries@`) were all dead on 2026-04-22. The Regulator migrated fully to the `@inforegulator.org.za` domain. If a future run finds the `.justice.gov.za` addresses re-listed on the contact page, treat it as a conflict and flag it.

**Decision logic:**
- Baseline address still listed → use it.
- Baseline address replaced → use the replacement; flag the change in your summary.
- Baseline address removed without replacement → drop it from the send list; flag the removal.
- Site lists a dedicated "Research", "Policy", or "Stakeholder Engagement" inbox not in the baseline → add it to Cc.
- Site unreachable → stop. Do not proceed on stale addresses. Return the error to Akhona.

Return the verified address list before composing.

### 2. Verify the Pansy Tlakula salutation

Use `WebFetch` on the Regulator's leadership page to confirm **Advocate Pansy Tlakula** is still the Chairperson. If the role holder has changed, adjust the salutation in your draft and flag the change. The letter on disk currently hardcodes her name — if replaced, edit your *send copy* (do not modify the source file without Akhona's sign-off).

### 3. Generate the PDF attachment

Run from the repo root:

```
pandoc docs/outreach/info-regulator-letter.md \
  -o docs/outreach/info-regulator-letter.pdf \
  --pdf-engine=xelatex \
  -V geometry:margin=2.5cm
```

If `pandoc` is not available, try `md-to-pdf`, `typora --export-pdf`, or `weasyprint`. If none are available, stop and ask.

Do **not** fabricate or attach the "one-page summary PDF" referenced in the letter footer — it does not exist. If and only if Akhona has authored it before you run, attach it as `docs/outreach/semantix-popia-onepager.pdf`. Otherwise, silently omit it.

---

## Composition

### Subject

```
Open-source POPIA compliance validation model — seeking feedback from the Information Regulator
```

### From

`akhona@automationarchitects.ai` — do not substitute.

### Body

Copy the letter body from `docs/outreach/info-regulator-letter.md` verbatim: everything between the horizontal rule following the preamble blockquote (line starting with "Dear Advocate Pansy Tlakula…") and the horizontal rule before the "Attachments to include" footer. Preserve section headings. Use plain text, not HTML.

Do not summarise, do not add an introductory line ("Please see attached…"), do not add a reply-handling instruction. The letter is the message.

### Signature

Append exactly this block — do not substitute, expand, or decorate:

```
Akhona Eland
Software engineer
akhona@automationarchitects.ai
https://github.com/labrat-akhona
```

### Attachments

- `docs/outreach/info-regulator-letter.pdf` (required)
- `docs/outreach/semantix-popia-onepager.pdf` (only if it exists at send time)

---

## Send / handback

If your environment has an authenticated, user-approved mail tool (e.g. an MCP tool with explicit send scope on Akhona's account), send using it. Return the send confirmation (message-id, timestamp, recipient list as actually delivered to).

If it does not, **stop after drafting**. Return:
- The composed subject
- The composed body
- The resolved recipient list (with any deltas from baseline flagged)
- A note that Akhona must send manually, and a one-line "to send, open your mail client and paste body + attach PDF"

Never send from a shared inbox, a generic Claude-session identity, or any address other than `akhona@automationarchitects.ai`.

---

## Hard rules

1. **No endorsement language.** If any reply in the thread asserts a working relationship with the Regulator, do not reinforce it. Forward to Akhona.
2. **Never use the phrase "POPIA-compliant"** — anywhere, including follow-ups and reply drafts.
3. **Written confirmation required** before changing any public artefact (model card, README, dataset card, demo Space) in response to regulator feedback. Verbal or implied guidance is insufficient.
4. **45-day publication silence.** Do not log, mention, or hint at this outreach in any public channel (Dev.to, Twitter/X, LinkedIn, Discord, GitHub issues/PRs, HF discussions, MCP registry notes). The lock releases when (a) the Regulator replies or (b) 45 calendar days elapse from send. If in doubt, silent.
5. **Do not reply autonomously to any inbound.** Forward to Akhona within 24 hours of arrival. Draft suggested replies — do not dispatch them.
6. **Do not modify `info-regulator-letter.md`** without explicit Akhona sign-off, even if the Regulator suggests wording changes.

---

## Follow-up cadence

Track the send timestamp in a file (`docs/outreach/regulator-send-log.jsonl`, one JSON line per event). Schedule:

| Trigger | Action |
|---|---|
| Send confirmed | Append `{"event": "sent", "ts": "...", "recipients": [...]}`. |
| 15 working days, no inbound | Draft a single follow-up in the same thread using the template below. Surface for Akhona's sign-off before dispatch. |
| 30 working days, still nothing | Draft a separate-thread escalation to a named Commissioner. As of 2026-04, candidate is **Commissioner Dr Tana Pistorius** (complaints & research). Verify role is current via <https://inforegulator.org.za/contact> before drafting. |
| 45 calendar days elapsed | Notify Akhona that publication silence is lifted. |

### 15-day follow-up template

```
Dear Advocate Tlakula and team,

I am following up on my email of <DATE> regarding the open-source
nli-popia-v1 model. I appreciate the Regulator's office receives a
high volume of correspondence and that a detailed response takes
time. Even a short acknowledgement would be valuable to me.

I remain available to travel to Pretoria or to meet virtually at
the Regulator's convenience.

Thank you again for your work.

Akhona Eland
```

No attachment, no new information, no tonal escalation.

---

## Reply-handling decision tree

For any inbound on this thread:

```
Inbound received
├─ Is it from the Regulator's office / *.justice.gov.za / *.inforegulator.org.za?
│  ├─ Yes → log, forward to akhona@automationarchitects.ai within 24h, draft but do not send a reply
│  └─ No  → log, forward to Akhona, do not engage
│
├─ Does it ask for a specific language change on the model card or artefacts?
│  → Draft the change as a diff against the source file, surface for Akhona's sign-off, do not commit
│
├─ Does it propose a meeting?
│  → Forward proposed slots. Do not accept, decline, add attendees, or share any meeting link.
│
├─ Does it assert the tool should not be distributed?
│  → Flag urgent. Forward within 2 hours of detection. Hold all outbound until Akhona responds.
│
├─ Is it from a journalist?
│  → Do not engage. Forward with a "press contact" flag.
│
└─ Is it an out-of-office auto-reply?
   → Log only. No forward. No follow-up count triggered.
```

Draft replies in the same tone as the letter: plain, non-promotional, specific. Never speak on Akhona's behalf in a way that commits him — use *"I will pass this to Akhona and he will respond directly"* as the holding reply.

---

## Success signals

Any of the following is a successful outcome — record and notify Akhona:

- Specific wording corrections on the model card or README
- An explicit or implicit meeting offer (even informal)
- A referral to another desk or officer
- A stated regulator posture on tools of this kind (even cautious)
- A "received and noted" acknowledgement with any officer's name attached

No reply is the default and is not a failure.

---

## What you return to Akhona

At every handoff point (preflight complete, draft ready, inbound received, follow-up scheduled, 45-day unlock reached), return:

1. A one-line status
2. Any diffs against baseline (addresses, salutation, attachments)
3. Explicit next-action with owner (you or Akhona)
4. Any rule from this brief that nearly triggered but did not, with one-line rationale
