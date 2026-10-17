# Schedule — Sat 17 Oct 2026 (hacking 10:00–16:00)

| Time | Block | Alex + hx-builder | Teammate A + hx-guard | Teammate B + hx-data / hx-redteam | Must be true by end |
|---|---|---|---|---|---|
| 10:20–10:45 | Kickoff | Scaffold Vite+Vue+TS, Vercel link, `api/explain.ts` stub | Threat model (1 page, `tasks/threat-model.md`) | `hx-data`: dataset schema + first 10 destinations | Repo runs locally, schema agreed |
| 10:45–12:15 | Core loop | `score.ts` + tests, inputs UI, top-3 cards | `validate.ts`: input bounds, free-text cap, enum moods | `hx-data`: 20 destinations, `npm run check:data` passes | **11:30:** ranking works on real data. **12:15:** committed + Vercel preview URL |
| 12:30–13:15 | Lunch / networking | Talk to Skyscanner judges (questions from `hx-pitch`) | same | same | 2 judge insights noted in `pitch/judges.md` |
| 13:15–14:45 | The flip | `/api/explain` live + fallback, "what changed" diff, polish the demo moment | Prompt-injection defence + output grounding in `api/explain.ts` | `hx-redteam`: injection + sanity suite, files findings | Demo moment works end to end on the deployed URL |
| **14:45** | **FEATURE FREEZE** | | | | |
| 14:45–15:20 | Harden | Fix only red-team blockers | Re-test fixed issues | `hx-pitch`: script v1 | Zero open blockers |
| 15:20–15:45 | Ship | Final deploy, record 60s backup video of the demo | Verify key not in bundle (`grep` the build) | Rehearse twice, time it | Pitch ≤ 3 min, backup video saved |
| 15:45–16:00 | Submit | Submit (Alex approves text) | | | Submitted before 16:00 |
| 16:00–17:00 | Judging | Demo to judges | | | |

Checkpoint reminders fire at 11:28, 12:12, 13:13, 14:43, 15:18, 15:43, 15:52.

## Cut list (in order, if behind)
1. Map view → 2. Time Machine chips → 3. "what changed" diff (keep simple re-rank) →
4. Live LLM (ship templated explanations only). Never cut: dataset, scoring, deployed URL, pitch.
