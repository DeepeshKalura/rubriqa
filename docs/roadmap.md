# Roadmap

Living list of the next moves for `rubriqa`, in priority order. Each item
links to its PR once it ships. Anything not on this list is out of scope
for the dev phase (see ADR-0001).

## Open

### 1. Scoring iteration loop — `feat/scoring-iteration-loop`
**Goal.** Measure and reduce the gap between `Entry.human_score` and the
LLM's score, without touching the critical path.

**Adds.**
- `scripts/score_replay.py` — re-runs the rubric on entries where
  `human_score IS NOT NULL` using the latest prompt + N-shot examples
  drawn from past human decisions (per-theme).
- `scripts/bench_scoring.py` — computes Cohen's κ + MAE per criterion
  per theme, writes `reports/scoring_bench_<date>.json`.
- Make targets `score-replay` and `bench-scoring`. Both run inside Docker.
- A `golden_set` SQL view in `app/services/golden_set.py` so the scripts
  read from the same source.

**Success criterion.** `make bench-scoring` runs on the seeded demo
program in < 30 s and produces a JSON file the owner can read.

### 2. Auth flag — `feat/auth-flag`
**Goal.** Add user accounts without committing to them. The flag stays
off by default; turning it on lights up login, bcrypt, sessions, 403 on
every router.

**Adds.**
- `auth_enabled: bool = False` in `app/config.py`.
- `app/auth.py` — Starlette `SessionMiddleware`, bcrypt-hashed env-driven
  bootstrap admin (`AUTH_ADMIN_USER`, `AUTH_ADMIN_PASSWORD_HASH`).
- `users` table migration (only created when the flag is on; behind a
  `/healthz/feature-flags` report).
- `POST /auth/login`, `POST /auth/logout`, `GET /auth/me`.
- `require_user` dependency. Wired into the existing routers only when
  `auth_enabled=True`.
- `.env.example` rows for the new env vars.
- Unit + integration test that toggles the flag on/off.

**Success criterion.** With `AUTH_ENABLED=false`, every existing test
still passes. With `AUTH_ENABLED=true`, an unauthenticated request gets
`403`, a valid login gets through, and the seed user can sign in.

### 3. Promote entries — `feat/promote-route`
**Goal.** Turn an `approved` review sample into a state transition on the
parent entry, with one button on the entries page.

**Adds.**
- `app/services/promotion.py` — pure function that:
  - reads the latest approved ReviewSample for an entry,
  - sets `entries.state = APPROVED` (or `REJECTED` if the sample is
    rejected),
  - copies `human_score` to the canonical score column,
  - logs the transition (structlog `entry_promoted` / `entry_rejected`).
- `POST /entries/{id}/promote` route. Idempotent.
- **Promote** button on `app/templates/entry_detail.html`, gated on
  `review_status == approved`.
- Tests: happy path, double-promotion is no-op, rejection path, no sample
  yet returns 409.

**Success criterion.** A reviewer can approve a sample in the batch UI,
then on the entry detail click Promote and the chip turns green without
re-reading the page.

## Closed

- **Drop legacy exam pipeline (ADR-0001).** Closed in `e03ab2d`.
