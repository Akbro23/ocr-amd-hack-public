#!/usr/bin/env bash
# Assemble the submission image = rocm/pytorch base + the exact layers build_image.sh tested and pushed.
# Runs on GitHub Actions (the notebook's proxy blocks Docker Hub). The base is referenced inside
# Docker Hub, never downloaded, so its layers stay byte-identical (the graders' base check).
#   LAYERS=ghcr.io/<github-user>/ocr-layers:v1 IMAGE=docker.io/<user>/<name>:v1 tools/assemble.sh
set -euo pipefail
: "${LAYERS:?}" "${IMAGE:?}"
BASE=docker.io/rocm/pytorch@sha256:3174cb7061d94c427da96c0edef4adea28046fa3f3b2ff3948dc4e995665ff8c
REPO=$(cd "$(dirname "$0")/.." && pwd)
D=$(mktemp -d -p "${RUNNER_TEMP:-/tmp}")

digests() { crane manifest "$1" | python3 -c "import json,sys; print('\n'.join(l['digest'] for l in json.load(sys.stdin)['layers']))"; }

echo "== fetch tested layers from $LAYERS"
args=()
for d in $(digests "$LAYERS"); do
  crane blob "${LAYERS%:*}@$d" > "$D/${d#sha256:}.tar.gz"   # compressed as pushed: same digest on re-append
  args+=(-f "$D/${d#sha256:}.tar.gz")
done
du -sh "$D"

echo "== append to base, push $IMAGE"
crane append -b "$BASE" "${args[@]}" -t "$IMAGE"
crane config "$IMAGE" | python3 "$REPO/tools/image_config.py" | crane edit config "$IMAGE" -t "$IMAGE"

echo "== verify"
want=$(printf '%s\n%s' "$(digests "$BASE")" "$(digests "$LAYERS")")
[ "$(digests "$IMAGE")" = "$want" ] || { echo "FAIL: image layers are not exactly base + tested layers"; exit 1; }
echo "layers = $(digests "$BASE" | wc -l) base (identical) + $(digests "$LAYERS" | wc -l) tested: ok"
crane config "$IMAGE" | python3 -c "
import json, sys
c = json.load(sys.stdin)['config']
env = dict(e.split('=', 1) for e in c['Env'])
print('Cmd', c['Cmd']); print('Healthcheck', c['Healthcheck']['Test']); print('\n'.join(c['Env']))
assert c['Cmd'] == ['python3', '/app/server.py'], 'wrong Cmd'
assert '/opt/ocr/deps' in env.get('PYTHONPATH', '').split(':'), 'PYTHONPATH misses /opt/ocr/deps'
assert env.get('OCR_MODEL') == '/models/Qwen3.5-9B', 'wrong OCR_MODEL'
assert env.get('PATH', '').startswith('/opt/venv/bin'), 'python3 would not be the venv'
print('config: ok')"
echo "pushed $IMAGE"
