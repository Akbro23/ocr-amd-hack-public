#!/usr/bin/env bash
# Build and test our layers on the notebook, then push them (alone) to ghcr.io.
# The notebook's egress proxy blocks Docker Hub, so the final image is assembled by the
# GitHub Actions workflow "assemble" (tools/assemble.sh), which reuses these exact layers.
#
#   tools/build_image.sh                                        # first: installs crane, then stops
#   crane auth login ghcr.io -u <github-user> --password-stdin   # classic GitHub token with write:packages
#   LAYERS_REPO=ghcr.io/<github-user>/ocr-layers TAG=v1 tools/build_image.sh
#   then on the laptop: gh workflow run assemble -f tag=v1
set -euo pipefail

REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a   # Qwen/Qwen3.5-9B
WEIGHTS=${WEIGHTS:-/workspace/models/Qwen3.5-9B}
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=/opt/venv/bin/python
DEPS=/opt/ocr/deps
S=/root/stage   # container disk: plenty of room, wiped with the session
export PATH=/root/bin:$PATH

command -v crane >/dev/null || { mkdir -p /root/bin; curl -sL https://github.com/google/go-containerregistry/releases/download/v0.22.1/go-containerregistry_Linux_x86_64.tar.gz | tar -xz -C /root/bin crane; }
command -v uv >/dev/null || pip install -q uv
: "${TAG:?crane is installed in /root/bin; now log in to ghcr.io and set TAG=v1}"
LAYERS=${LAYERS_REPO:?set LAYERS_REPO=ghcr.io/<github-user>/ocr-layers}:$TAG
grep -rqs "$REVISION" "$WEIGHTS/.cache" || { echo "weights in $WEIGHTS are not revision $REVISION"; exit 1; }

rm -rf "$S" && mkdir -p "$S"
layer() {  # layer <out.tar> <root dir>: reproducible tar of everything under root
  tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
      --exclude=.cache --exclude=__pycache__ -C "$2" -cf "$1" $(ls -A "$2")
}

echo "== 1/4 deps layer -> $DEPS"
# torch comes from the base image: drop it (and everything only it needs) from the resolution
echo 'torch ; sys_platform == "never"' > "$S/no-torch.txt"
uv pip compile "$REPO/pyproject.toml" --python "$PY" --overrides "$S/no-torch.txt" --no-header -q -o "$S/reqs.txt"
uv pip install -q --python "$PY" --no-deps --target "$S/deps$DEPS" -r "$S/reqs.txt"
layer "$S/deps.tar" "$S/deps"

echo "== 2/4 test staged deps on this GPU (base torch + staged packages only)"
PYTHONPATH="$S/deps$DEPS" "$PY" -c "import transformers, accelerate; assert transformers.__file__.startswith('$S'), transformers.__file__"
(cd "$REPO" && PYTHONPATH="$S/deps$DEPS" OCR_MODEL="$WEIGHTS" "$PY" tools/eval.py data/samples)

echo "== 3/4 weights + code layers (one layer per weight file: ghcr.io caps layers at 10 GB)"
TARX=(--sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner --transform 's|^|models/Qwen3.5-9B/|S' -C "$WEIGHTS")
i=0
for f in "$WEIGHTS"/*.safetensors; do
  tar "${TARX[@]}" -cf "$S/w$((++i)).tar" "$(basename "$f")"
done
tar "${TARX[@]}" -cf "$S/w0.tar" $(cd "$WEIGHTS" && ls -A | grep -v -e '\.safetensors$' -e '^\.cache$')
mkdir -p "$S/code/app/input" "$S/code/app/output"
cp "$REPO"/app/*.py "$S/code/app/"
layer "$S/code.tar" "$S/code"
du -ch "$S"/*.tar

echo "== 4/4 push layers to $LAYERS"
args=(-f "$S/deps.tar" -f "$S/w0.tar")
for f in "$S"/w[1-9]*.tar; do args+=(-f "$f"); done
args+=(-f "$S/code.tar")
crane append "${args[@]}" -t "$LAYERS"
echo "our layers (uncompressed): $(du -ch "$S"/*.tar | tail -1 | cut -f1); base <= 36G; limit 60 GiB"
echo "pushed $LAYERS -> now on the laptop: gh workflow run assemble -f tag=$TAG"
