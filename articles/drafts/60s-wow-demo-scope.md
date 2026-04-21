# 60-second-wow demo scripts — scope proposal

**Status:** scope-for-review. No code changed yet. The three scripts below would ship as new subcommands on the existing `semantix` CLI (entry point already exists at `semantix/cli.py`). Each one is purely additive — no changes to existing APIs, no new runtime dependencies.

**Goal:** engineer the first-60-seconds "oh" moment for a new user. Someone who `pip install`s semantix and runs one of these commands should feel the advantage immediately, not after reading a benchmark blog.

## Why these three

From the earlier strategy discussion: the three things that most need to be *felt* (not argued) to convert a new user are

1. **Speed** — the whole thing is fast on first contact
2. **Determinism** — same input gives the same score, forever
3. **Auditability** — every validation leaves a receipt that's actually interesting to look at

Each demo targets one of these. All three work offline, zero API keys, on a fresh install.

---

## 1. `semantix demo` — the speed + correctness surprise

**Goal:** show that semantix validates real outputs against a real intent in well under a second, with visible pass/fail/receipt output, on a laptop.

**What it does:**

- Defines one built-in intent (`"The response must acknowledge the customer's complaint and propose a concrete next step"`)
- Holds 8–12 realistic customer-service replies as fixtures (5 that should pass, 5 that should fail, a couple of edge cases)
- Runs them all sequentially
- Prints a coloured table with columns: `#`, `output (truncated)`, `score`, `verdict`, `latency`
- Prints a footer with: total wall-clock time, p50/p95/p99 latency, mean score for passes vs fails
- Writes an audit log (`./semantix-demo-audit.jsonl`) and prints the hash of the final certificate

**Honest framing in the CLI output:**

```
Ran 12 validations in 487 ms on CPU.
Latency p50 18.2 ms · p95 41.4 ms · p99 52.8 ms.
Audit chain: 12 certificates, final hash abcd1234…
Receipt at ./semantix-demo-audit.jsonl
```

No API keys. No network after the first-run model download (one-time, progress-bar visible).

**Scope boundaries:**

- Does NOT compare against LLM-as-judge. Doing that requires an API key, which breaks the "zero friction" premise. We keep that comparison for the benchmark repo where it belongs.
- Does NOT claim "25× faster" or any ratio in the demo output — the user sees the real numbers on their machine and draws their own conclusion.
- Does NOT modify or create files outside `./semantix-demo-audit.jsonl` (prints the path first, asks to continue if non-interactive flag is absent).

**New code surface:**

- `semantix/cli.py` — add `demo` subparser and `_run_demo(args)` function
- `semantix/_demo_data.py` — a small module holding the fixture intent and outputs
- Tests in `semantix/tests/test_cli_demo.py` — mock the judge, assert the CLI produces expected pass/fail counts and exits cleanly

**Estimated effort:** ~2-3 hours including tests.

---

## 2. `semantix prove` — the determinism demo

**Goal:** show that running the same validation 100 times produces 100 identical scores, visibly, in under 3 seconds.

**What it does:**

- Takes `--text` and `--intent` arguments (or uses a sensible default if neither is given)
- Runs `N=100` evaluations (override with `--n`)
- Collects all scores
- Prints the result as: `Score: 0.94217 — 100/100 runs agreed to 5 decimal places. Total wall-clock: 1.84 s.`
- If any variance is detected (which shouldn't happen with QuantizedNLI), prints a per-run histogram and exits non-zero.

**Contrast framing (without requiring an API key):**

The CLI output includes a short "why this matters" note:

```
Determinism verified: 100/100 identical scores.
The same check with an LLM-as-judge (e.g. gpt-4o-mini at temperature=0) typically
produces 3–7 distinct scores across 100 runs. Temperature=0 does not guarantee
determinism for LLM APIs — only for deterministic local models like this one.
```

That's a verifiable external claim we can footnote in docs if needed. We're not making that call in the demo; we're just stating the fact and letting the user run their own comparison.

**Scope boundaries:**

- Does NOT call any remote API.
- Does NOT modify files.
- Accepts `--text` / `--intent` via CLI arguments, stdin, or falls back to a built-in example.

**New code surface:**

- `semantix/cli.py` — add `prove` subparser and `_run_prove(args)` function
- Tests in `semantix/tests/test_cli_prove.py` — assert 100/100 identical scores with a mocked judge; assert non-zero exit with a jittering mock

**Estimated effort:** ~1-2 hours including tests.

---

## 3. `semantix verify <audit-log>` — the receipt beautifier

**Goal:** make the audit trail visibly interesting, inspectable from the terminal, and trivially shareable (screenshotable).

**What it does:**

- Takes a path to a JSONL audit log produced by `AuditEngine`
- Walks the chain, verifying each hash link
- Prints a coloured summary:

```
Audit chain: ./semantix-demo-audit.jsonl
✓ 12 certificates
✓ Chain intact (12/12 hash links verified)
First: 2026-04-21T14:22:10.004Z
Last:  2026-04-21T14:22:11.218Z

Certificates by verdict: 7 pass · 5 fail
Distinct intents: 1
Distinct judges: 1 (QuantizedNLIJudge 2026-01-03)

Head hash: abcd1234ef5678…
```

- With `--verbose`: prints each certificate in a compact JSON-LD box with highlighted hash links.
- On tamper detection: highlights the broken link in red, shows the expected vs actual hash, exits non-zero.

**Why this lands:**

Screenshottable output is its own marketing channel. A compliance officer or AI governance lead looking at a verified audit chain terminal output will either (a) forward it to their colleague or (b) send it to their procurement team. Both are conversion events we don't have to pay for.

**Scope boundaries:**

- Read-only. Never modifies the audit log.
- Accepts JSONL files produced by the current `AuditEngine` — no format changes.
- On format mismatch (old-format or corrupted file), prints a clear error and exits non-zero.

**New code surface:**

- `semantix/cli.py` — add `verify` subparser and `_run_verify(args)` function
- `semantix/audit/verify.py` — pure function `verify_chain_file(path) -> VerifyResult` with no I/O side effects beyond reading; CLI wraps it for output
- Tests in `semantix/tests/test_cli_verify.py` — three fixtures: clean chain, tampered chain, malformed file

**Estimated effort:** ~3-4 hours including tests and fixtures.

---

## Total effort estimate

~6-9 hours of focused work to ship all three with tests. Could be split across two sessions.

## Rollout order

1. `prove` first — smallest, highest delight-per-line-of-code ratio.
2. `verify` second — unlocks the screenshot-the-receipt marketing moment.
3. `demo` third — depends on having polished the first two because it uses both implicitly.

## Documentation changes that ride along

- Update `docs/getting-started.md` — add a "First 60 seconds" section that just says: run `semantix demo`, `semantix prove`, `semantix verify`. Three commands, three surprises.
- Add a one-line mention in the README's opening code block section.
- Update the docs home to include a terminal-recording GIF of `semantix demo` (recorded with `asciinema` + `agg` — no proprietary tooling).

## Open questions for your decision

1. **Fixture data for `semantix demo`** — I'd ship ~12 realistic customer-service outputs. These shouldn't resemble any real customer data. Options: write them myself (cleanest), use a tiny public snippet from the customer_support benchmark (internal consistency), or ship two or three built-in demo intents the user picks between (more surface). Recommend: write ~12 from scratch, customer-service domain, ship them as `semantix/_demo_data.py` with a comment marking them as synthetic fixtures.
2. **Should `semantix demo` write to the current directory or to a temp directory?** Writing to CWD is more discoverable (user sees the file); writing to a temp dir is tidier. Recommend: CWD, but print the path prominently and add a `--out` flag to override.
3. **`semantix prove` default — should it use a built-in text/intent or require args?** Requiring args makes the demo friction longer. A built-in default lets the user literally type `semantix prove` and see the determinism moment in 3 seconds. Recommend: built-in default, with `--text` / `--intent` to customise.
4. **`--no-color` flag?** Some terminals and CI systems mangle ANSI codes. Worth adding a `NO_COLOR=1` env var check (per [no-color.org](https://no-color.org/)) and a `--no-color` flag. Recommend: yes, both.

When you're ready, give the go-ahead and I'll ship in the order above, committing each subcommand + its tests separately so any one of them can be reverted cleanly.
