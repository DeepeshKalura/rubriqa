# TASTE.md — what this project's owner actually values

Taste is revealed by what a person asks for, questions, rejects, and repeats.
This file records each milestone of the build with a score against that
evolving taste profile. New entries are appended automatically at every
milestone (see AGENTS.md step 10).

## Preference profile (derived from the owner's own words)

| # | Preference | Evidence |
|---|---|---|
| P1 | **Docker-first, one small command to run everything** | "make the plan use docker for everything" |
| P2 | **Feature flags, never permanent hardcoding** | "no hybrid, use a feature flag, either one or two we can choose ... feature flag is perfect" |
| P3 | **Evidence over claims: validation data, benchmarks, screenshots** | "validation data generation is important", "use a skills you can use" (browser verification) |
| P4 | **No over-engineering; local, resource-conscious** | explicit 16 GB office-laptop constraint, no Redis/K8s |
| P5 | **Bilingual Hindi + English is a first-class reality** | "Primary languages: Hindi, English" |
| P6 | **Self-documenting project memory (AGENTS.md + TASTE.md)** | "create agents.md file as well taste.md file ... each command add and give a score" |
| P7 | **Real end-to-end flow, not disconnected examples** | brief: "not as a prototype or a collection of disconnected examples" |

## Scoring rubric (0–5 per entry)

- +1 works end-to-end and is verifiable in Docker
- +1 respect for feature flags / provider boundaries
- +1 evidence produced (test, benchmark, screenshot)
- +1 simplicity (nothing unnecessary added)
- +1 documentation/taste memory updated
- penalties for hardcoded config, unverified claims, skipped evidence

## Log

| Date | Milestone / command | Owner asked | Score | Tags | Lesson |
|---|---|---|---|---|---|
| 2026-10-06 | Recon: verified MiniMax key, models, vision, JSON mode; port conflicts 8080/5432-5434 | plan first | 5/5 | recon, evidence | Recon before code prevented wrong model/port choices |
| 2026-10-06 | Plan approved with amendments (flags not hybrid, validation data, AGENTS.md/TASTE.md, skill-based UI verification) | use docker for everything, feature flags, evidence | 5/5 | docker, flags, docs | Owner consistently prefers reversible decisions over clever defaults |
| 2026-10-06 | `docker compose build web` + `make`-style flow scaffolded; schema 25 tables migrated; `/healthz`, `/readyz` green | — | 4/5 | docker, db, evidence | Stale image vs bind-mount confused one debug cycle; source bind mounts added for dev speed |
| 2026-10-06 | Pipeline E2E with real MiniMax (26 docs, 31 pages, 20 answers, 47 scores, 0 failed pages) | evidence over claims | 5/5 | e2e, minimax, bilingual | Live Hindi+English extraction worked first try after JSON-robustness fix |
| 2026-10-06 | Reader benchmark rebuilt around answer-recall (CER alone punished inserted questions) | honest numbers | 4/5 | benchmark, evidence | First metric was misleading; owner's "validation data is important" paid off |
| 2026-10-06 | 20k submissions seeded; API p50 10-126 ms, 12-16 KB payloads, containers 83/83/182 MB | resource-conscious | 5/5 | scale, evidence | Set-based SQL + pagination held; no browser ever saw a full table |
| 2026-10-06 | Feature flags exercised: reader modes, repair, scoring strategies, Google Forms mock — all switchable at runtime | feature flags | 5/5 | flags | Settings page edits DB overrides; .env remains the baseline |
| 2026-10-06 | 96 tests + 4 live tests green; failure/retry/sweeper paths proven via simulated stale heartbeat | testing | 5/5 | tests | Retry UI closed the loop on a real failed job |
| 2026-10-06 | UI verified by Playwright: 37 checks, 16 screenshots incl. failed-job state and post-restart persistence | evidence, docker | 5/5 | ui, screenshots | Screenshots are real states, not mockups |
| 2026-10-06 | README + architecture + setup + limitations written with measured numbers | honest limitations | 5/5 | docs | Limitations list states exactly what was NOT tested |
