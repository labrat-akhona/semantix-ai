# semantix-ai Pro Tier — Design Spec

## Goal

Turn semantix-ai from a free open-source tool into a revenue-generating product with a cloud dashboard, team features, and CI integration — while keeping the core library free and local.

## Architecture

**Free tier (what exists now):**
- Local NLI/embedding/quantized judges
- All integrations (pytest, Guardrails, LangChain, etc.)
- assert_semantic, validate_intent, training collector
- Everything runs offline, no data leaves the machine

**Pro tier (cloud dashboard):**
- Semantic test trend tracking over time
- Team-wide intent drift detection
- Slack/email alerts when production outputs fail checks
- CI badge showing semantic test health
- Centralized training data collection across environments

## How It Works

```
[Your CI / pytest]
     |
     | --semantic-report-json → semantic-results.json
     |
     v
[semantix upload]  ← new CLI command
     |
     | POST /api/v1/results (API key auth)
     |
     v
[semantix.cloud]
  - Dashboard (React/Next.js)
  - API (FastAPI)
  - DB (PostgreSQL)
  - Alerts engine (Slack webhook, email)
```

### Key Insight

The JSON report from `--semantic-report-json` already contains everything needed:
- test name, intent, pass/fail, score, reason, duration
- We just need a place to send it and visualize trends

### New Components

1. **`semantix upload` CLI command** — reads `semantic-results.json`, POSTs to cloud API
2. **Cloud API** — FastAPI, receives results, stores in Postgres, triggers alerts
3. **Dashboard** — shows trends, failing intents, score distributions, team view
4. **GitHub Action update** — add `upload: true` option to auto-upload after tests

### Pricing Ideas

| Plan | Price | Features |
|------|-------|----------|
| Free | $0 | Everything local, unlimited |
| Team | $29/mo | Dashboard, 5 seats, 30-day history, Slack alerts |
| Business | $99/mo | Unlimited seats, 1-year history, custom judges, SSO |

### MVP Scope (what to build first)

1. `semantix upload` CLI command
2. Cloud API that receives and stores results
3. Simple dashboard showing last 30 days of results per intent
4. One alert type: Slack webhook when a previously-passing intent starts failing

### Tech Stack

- **API:** FastAPI + PostgreSQL (deploy on Railway or Fly.io)
- **Dashboard:** Next.js + shadcn/ui (deploy on Vercel)
- **Auth:** API key per team (simple, no OAuth initially)
- **Alerts:** Slack incoming webhook

### What NOT to Build Yet

- User accounts / OAuth / SSO
- Custom judge hosting
- Training data management in cloud
- Billing integration (start with manual invoicing)
