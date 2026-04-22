# Letter to the SA Information Regulator

> **Purpose:** introduce `nli-popia-v1`, a locally-hosted open-source model for validating LLM outputs against POPIA clauses. Ask for (a) a review of the intended-use framing, (b) feedback on the clause hypotheses, and (c) whether the Regulator's office would be open to a working meeting.
>
> **Not asking for:** endorsement, certification, or any claim that the model is "POPIA-compliant" — those are determinations only the Regulator can make.
>
> **Recipients (verify before sending):**
> - Information Officer, Information Regulator South Africa: [inforeg@justice.gov.za](mailto:inforeg@justice.gov.za)
> - Complaints / general queries: [complaints.IR@justice.gov.za](mailto:complaints.IR@justice.gov.za)
> - Copy: [enquiries@justice.gov.za](mailto:enquiries@justice.gov.za)
>
> **Before sending:** verify all three email addresses at <https://inforegulator.org.za/> — they move from time to time. If the Regulator publishes a new address after 2026-04, use that.

---

**Subject:** Open-source POPIA compliance validation model — seeking feedback from the Information Regulator

Dear Advocate Pansy Tlakula and the Information Regulator team,

My name is Akhona Eland. I am a South African software engineer. I am writing to introduce an open-source research artefact that I believe sits adjacent to the Regulator's remit, and to respectfully ask for guidance on how to frame it responsibly.

## What it is

Over the past month I fine-tuned and released a small Natural Language Inference (NLI) model — [`nli-popia-v1`](https://huggingface.co/labrat-aiko/nli-popia-v1) — that scores whether a given English text (for example, a signup-flow message, a breach-notification email, or a customer-support reply) is consistent with seven canonical POPIA clauses:

1. Consent
2. Minimality / purpose limitation
3. Security safeguards
4. Breach notification
5. Cross-border transfers
6. General lawful processing
7. Data subject rights

The model runs entirely on the user's own computer, in approximately 70 milliseconds per check, with no data leaving the machine and no API calls to foreign service providers. It is released under the Apache 2.0 licence, is free to use commercially, and ships with a release-gate requirement that no future version may regress on any of the seven clauses.

The model is published on Hugging Face with complete reproducibility:

- Model: <https://huggingface.co/labrat-aiko/nli-popia-v1>
- Training + evaluation dataset: <https://huggingface.co/datasets/labrat-aiko/popia-compliance-nli>
- Interactive demo: <https://huggingface.co/spaces/labrat-aiko/popia-judge-demo>
- Source library: <https://github.com/labrat-akhona/semantix-ai>

## What I am not claiming

I want to be very clear: I am **not** claiming this model makes any software "POPIA-compliant", and I am **not** claiming it replaces a Data Protection Impact Assessment, a Privacy Officer, or the Regulator's guidance. The model flags *semantic consistency* between a specific output and a specific clause at a specific numerical threshold. That is a narrower, technical determination than compliance, and it is the only claim I have designed the model to defend.

I have written the documentation, the model card, and the accompanying article with that distinction in mind. If the Regulator's office considers any of the language I have used to be too strong, I would welcome specific corrections and will incorporate them.

## What I am asking

Three things, in order of importance:

1. **A review of how the intended use is framed.** Are there phrasings in the model card, README, or demo that overstep — for example, any language that could be read as "this software guarantees POPIA compliance"? I would like to correct those before the artefact gains wider circulation.

2. **Feedback on the seven clause hypotheses.** Each clause is represented by a single-sentence "hypothesis" that the NLI model scores against. Those hypotheses are my engineering reading of POPIA, not a legal one. I would be grateful for any clause where the Regulator's office believes the hypothesis is materially misaligned with the Act.

3. **Whether the Regulator's office would be open to a short working meeting.** If a 30-minute call is feasible, I am happy to travel to the Regulator's offices in Pretoria or meet virtually. The agenda would be: walk through the model, show the evaluation methodology, discuss the intended framing, and hear any concerns the Regulator has about tools of this kind entering general use.

I understand the Regulator's office receives many enquiries. Any response — even "use this specific language instead" — would be genuinely valuable to me, and would make the artefact measurably safer for South African companies that might deploy it.

Thank you for the work you do. South Africa having an active, technically-engaged data protection regulator is something I am grateful for as both a citizen and a builder.

Yours sincerely,

**Akhona Eland**
Software engineer
[akhona@automationarchitects.ai](mailto:akhona@automationarchitects.ai)
<https://github.com/labrat-akhona>

---

**Attachments to include when sending:**

1. One-page summary PDF (next item to author) — describes the model, the training corpus, the 150-pair hash-pinned evaluation set, and the +29.6pp F1 lift over stock NLI. Keep it non-technical for the first half and technical for the second.
2. Link to the live demo Space (above) so a non-technical reviewer can try the tool in a browser without installing anything.

**Follow-up cadence:**

- If no response in 15 working days: one polite follow-up email, same thread.
- If no response in 30 working days: consider writing to an Information Regulator commissioner directly — Commissioner Dr Tana Pistorius handles complaints & research enquiries per [inforegulator.org.za/contact](https://inforegulator.org.za/contact) (verify current role before using).
- Do **not** publish the existence of this letter on social media or in the Dev.to article until a response is received or 45 days have elapsed — leading with "we wrote to the Regulator" without a reply invites misinterpretation.
