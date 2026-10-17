#!/usr/bin/env bash
# Lays out the hx- team in the current herdr workspace and starts each agent.
# Tab "Team": hx-orchestrator | hx-builder / checks shell
# Tab "Security & data": hx-guard | hx-data / hx-redteam
# Tab "Pitch": hx-pitch
set -euo pipefail
test "${HERDR_ENV:-}" = 1 || { echo "not inside herdr"; exit 1; }

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WS="${HERDR_WORKSPACE_ID}"
MODEL="${HX_MODEL:-sonnet}"

jget() { python3 -I -c "import sys,json;d=json.load(sys.stdin);print(eval('d'+sys.argv[1]))" "$1"; }

start() { # name pane
  local name="$1" pane="$2"
  herdr pane rename "$pane" "$name" >/dev/null
  herdr agent start "$name" --kind claude --pane "$pane" --timeout 60000 -- \
    --model "$MODEL" --permission-mode acceptEdits \
    --append-system-prompt "You are $name, a standing agent in the hackathon team. Your role file is $ROOT/.claude/agents/$name.md: read it now and follow it, together with BRIEF.md and SCHEDULE.md. Wait for tasks from hx-orchestrator or Alex. Keep replies short." \
    >/dev/null && echo "started $name in $pane" || echo "check $name in $pane (may be blocked on a trust/login dialog)"
}

tab() { herdr tab create --workspace "$WS" --cwd "$ROOT" --label "$1" --no-focus | jget "['result']['root_pane']['pane_id']"; }
split() { herdr pane split "$1" --direction "$2" --cwd "$ROOT" --no-focus | jget "['result']['pane']['pane_id']"; }

T1=$(tab "Team");            B=$(split "$T1" right);  C=$(split "$B" down)
T2=$(tab "Security & data"); D=$(split "$T2" right);  R=$(split "$D" down)
T3=$(tab "Pitch")
sleep 1

herdr pane rename "$C" checks >/dev/null
start hx-orchestrator "$T1"
start hx-builder "$B"
start hx-guard "$T2"
start hx-data "$D"
start hx-redteam "$R"
start hx-pitch "$T3"
echo "checks pane: $C"
