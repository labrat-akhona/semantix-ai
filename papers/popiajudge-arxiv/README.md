# POPIAJudge — arXiv preprint draft

Working draft of the preprint to be submitted to arXiv cs.CL.

## Files
- `main.tex` — full draft scaffold with abstract, intro, related work, method, experiments, companion artifacts, limitations.
- `references.bib` — citation database. All URLs verified to resolve at write time.

## Build
```bash
cd papers/popiajudge-arxiv
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## What still needs to be done before submission
- Fill in the `popia-instruct-v0` results section once that model finishes training (Phase 3 of the SA AI Compliance Stack ship list).
- A literature-edit pass for prose quality — the current draft prioritizes content correctness over style.
- Title and abstract polishing — the current versions are functional but not catchy.
- Acknowledgements section (none of this work was funded; the regulator outreach paper trail will be referenced in the camera-ready).
- A figure: macro F1 bar chart across stock / v1 / v2 on both holdouts.

## Submission checklist
- [ ] Resolve all `[Pending]` placeholders
- [ ] Run `chktex` for LaTeX hygiene
- [ ] Verify every `\cite` resolves
- [ ] Build clean PDF (no warnings on the final pass)
- [ ] Run a `pdftotext` sanity check that the abstract reads sensibly
- [ ] Submit to arXiv cs.CL with secondary categories cs.AI, cs.CY (computers and society)
