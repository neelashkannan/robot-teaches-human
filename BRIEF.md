# Brief

## Event
Edinburgh AI Hackathon — CompSoc x Edinburgh AI x UoE Informatics. Sponsors: Wordsmith, Skyscanner.
Partners judge during the day. 09:00 doors · 10:00 hacking · 12:30 lunch/networking · 16:00 hacking
ends · 17:00 winners. That's **6 hours** of hacking (the shortlist page said 7 — it's 6).

## Challenge — "Flip the Script"
Take something we all assume has to work a certain way and turn it on its head. Pick something
familiar, challenge the assumptions behind it, build a completely different way of doing it.
"Don't just make it better. Flip the whole idea."

Open brief: not travel-only. But Skyscanner engineers are judging, so a travel flip lands in front
of people who know the domain.

## Chosen idea — FlipTrip AI (default; change here if the team picks otherwise)
- **Normally:** pick a destination → search flights → discover it's too expensive.
- **Flipped:** start from budget, days and mood → the app tells you where to go.
- **Demo moment:** change mood from "adventure" to "nightlife" (or budget +£50, "leave Friday") and the
  ranking visibly reshuffles, with the AI explaining *why* each place moved.
- **Judge objection to pre-empt:** Skyscanner "Everywhere" already exists. Answer: Everywhere sorts by
  fare; FlipTrip ranks on total trip cost (flight + stay + daily spend) against what you want the
  trip to *feel* like, and explains the trade-off.

Shortlist source: https://claude.ai/artifact/53vr8MG6UayqU2ffV4h2iL (backup: AI Travel Rescue).
Non-travel fallbacks if the team prefers: AI Student/Human Teacher, Anti-Autocomplete.

## Scope (MVP — must ship)
1. `data/destinations.json`: ~20 destinations from EDI, each with flight £ estimate, stay £/night,
   daily spend £, mood scores (adventure, nightlife, culture, relax, food 0–5), flight hours, tags.
   Estimates are labelled as estimates in the UI.
2. `src/lib/score.ts`: pure function `(inputs, destinations) → ranked list` with a per-destination
   breakdown (total cost, mood fit, why). Unit-tested.
3. UI: budget slider, days, mood chips, optional free-text "anything else". Top 3 cards + "what changed"
   diff when inputs change.
4. `/api/explain`: one LLM call explaining the top 3 against the inputs. Templated fallback.

## Stretch (only if MVP is committed and deployed)
- Travel Time Machine: "what if +£50 / leave Friday / one more day" chips.
- Map view. Share link with inputs in the URL.

## Out of scope
Live fares, booking, accounts, scraping Skyscanner.

## Stack
Vite + Vue 3 + TypeScript, Vercel (frontend + `api/explain.ts` function), Anthropic API
(`claude-haiku-5-5` for low demo latency; `claude-sonnet-5-5` if quality is lacking). Vitest for tests.
