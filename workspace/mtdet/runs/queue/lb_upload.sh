#!/usr/bin/env bash
# Leaderboard API calls with the team token from .env (run from the repo root). The token reaches curl through stdin
# (`-H @-`, fed by the printf builtin), so it is never in a process's argv, a command line or the output.
#   lb_upload.sh me          GET /api/me (quota, reset cards)
#   lb_upload.sh post <id> <zip>   POST /api/submissions/<id>/post (make it the team's entry)
#   lb_upload.sh <zip>       check the zip, then POST it to /api/submissions; the reply goes to <zip stem>.response.json
#                            and the Idempotency-Key to <zip stem>.idempotency_key (a retry reuses it: never counted twice)
set -uo pipefail
LB=https://web-production-c241e.up.railway.app
set -a; . ./.env; set +a
auth() { printf 'Authorization: Bearer %s\n' "$CS701_TOKEN"; }
# the token reaches grep as a pattern file from the printf builtin, never in grep's argv (visible to ps)
no_token() { if grep -qF -f <(printf '%s\n' "$CS701_TOKEN") "$1"; then echo "TOKEN FOUND IN $1: not printing it"; return 1; fi; }

if [ "$1" = me ] || [ "$1" = list ]; then
    out=$(mktemp)
    path=$([ "$1" = me ] && echo /api/me || echo /api/submissions)
    code=$(auth | curl -sS -o "$out" -w '%{http_code}' "$LB$path" -H @-)
    echo "HTTP $code"; no_token "$out" && cat "$out"; echo; rm -f "$out"
    exit 0
fi

if [ "$1" = post ]; then  # lb_upload.sh post <submission id> <zip>: the reply goes to <zip stem>.post.json
    out=${3%.zip}.post.json
    code=$(auth | curl -sS -o "$out" -w '%{http_code}' -X POST "$LB/api/submissions/$2/post" -H @-)
    echo "$(date '+%F %T') HTTP $code post $2"; no_token "$out" && cat "$out"; echo
    exit 0
fi

ZIP=$1; STEM=${ZIP%.zip}
uv run python workspace/sarmtl/check_submission.py "$ZIP" --split val | tail -n 1
[ "${PIPESTATUS[0]}" = 0 ] || { echo "check_submission.py failed: not uploading $ZIP"; exit 1; }
[ -s "$STEM.idempotency_key" ] || uuidgen > "$STEM.idempotency_key"
code=$(auth | curl -sS -o "$STEM.response.json" -w '%{http_code}' -X POST "$LB/api/submissions" -H @- \
    -H "Idempotency-Key: $(cat "$STEM.idempotency_key")" -F "file=@$ZIP")
echo "$(date '+%F %T') HTTP $code $ZIP"
no_token "$STEM.response.json" && cat "$STEM.response.json"; echo
