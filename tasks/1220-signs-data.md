# 1220 signs data — hx-data

**Goal:** `data/signs.json` with 5 to 6 verified one-handed BSL signs the Amazing Hand can show, and a
check script.

**Read first:** `BRIEF.md` (Honest scope), `.claude/agents/hx-data.md`, `hand/vec.py`.

**Files allowed:** `data/`, `scripts/check_signs.py`. Nothing else.

**Do**
1. Research with WebSearch/WebFetch on BSL SignBank (https://bslsignbank.ucl.ac.uk/) and the BDA /
   Signature sources. Choose one-handed signs where the handshape is the defining part and a
   4-finger hand (no little finger, no wrist) can show it. Candidate families: number handshapes,
   simple single-handed words with a clear handshape. Do NOT use fingerspelling letters (two-handed).
2. For each, record `source_url`, `status`, and the handshape in words per finger in `notes`
   (extended / curled / half; thumb position; spread). Convert to the 8-number `target` by the
   rules in `hand/vec.py`. Mark anything you cannot confirm `unverified`.
3. `scripts/check_signs.py` as specified in your role file. Pose-loading check can come later when
   `hand/teacher.py` lands (tell the orchestrator).
4. Also write `data/sources.md`: one line per sign with the page you used, and a list of signs you
   rejected and why (for example needs the little finger, two hands, or movement).

**Done when:** check script passes, at least 5 non-`unverified` signs, each with a source URL.
Reply with 5 lines max: count, statuses, check output, anything you could not verify.
**Time box:** until 13:10.
