# ocr-amd-hack

AMD AI Challenge, Mini-Challenge 2 (OCR). Rules: `overview.pdf`. Model: Qwen3.5-9B.

Edit on the laptop, push to GitHub, run everything on the notebook. No installs on the laptop.

## Notebook: first session

```bash
P=/workspace
cd $P
# username: <github-user>, password: a fine-grained token for this repo only (Contents: read/write)
git -c credential.helper="store --file=$P/.git-credentials" clone https://github.com/<github-user>/ocr-amd-hack.git
git config --global --add safe.directory $P/ocr-amd-hack   # /workspace reports another owner
cd ocr-amd-hack
git config credential.helper "store --file=$P/.git-credentials"   # saved in the clone, survives sessions
git config user.name "<name>"; git config user.email "<email>"
```

## Notebook: every session

```bash
P=/workspace
git config --global --add safe.directory $P/ocr-amd-hack   # home is wiped each session
cd $P/ocr-amd-hack && git pull
pip install uv
PY=/opt/venv/bin/python   # the base image's venv, where ROCm torch lives (--system picks the wrong Python)
uv pip freeze --python $PY | grep -E '^(torch|torchvision)==' > constraints.txt
uv pip install --python $PY -r pyproject.toml --extra dev -c constraints.txt
export OCR_MODEL=$P/models/Qwen3.5-9B
```

## Notebook: once, after the first install

```bash
hf download Qwen/Qwen3.5-9B --local-dir $P/models/Qwen3.5-9B   # ~19 GB; hf comes with transformers
```

## Phase 0 check

```bash
rocm-smi                           # which GPU did we get?
python tests/test_text.py          # cleanup + normalization -> "ok"
python tools/extract_samples.py    # data/samples + labels.json (needs the challenge's overview.pdf in the repo root)
python tools/eval.py data/samples  # pass count, seconds/image, peak VRAM
```
Go if images take about 5 s or less each and peak VRAM is at least 1 GiB. Otherwise switch `OCR_MODEL` to Qwen3-VL-8B-Instruct.

## Harness check (no Docker needed)

Same calls the graders make: a fresh `python3 app.py` per image, talking to the warm server.
```bash
python app/server.py &            # wait for "ready in ..."
for f in data/samples/*; do [ "${f##*.}" = json ] && continue
  time OCR_OUT=output python3 app/app.py --input-image "$f"; done
cat output/*.json; echo
kill %1
```
Each call should take about 1 s, and every image should have a `<name>_output.json`.

## Build and push the image

Two halves, because the notebook's proxy blocks Docker Hub:
1. **Notebook** (`tools/build_image.sh`): builds our layers (deps in `/opt/ocr/deps`, weights in `/models`,
   code in `/app`), runs the eval with exactly those packages, pushes the layers alone to ghcr.io.
2. **GitHub Actions** (`assemble` workflow, `tools/assemble.sh`): puts those exact layers on top of the
   ROCm base inside Docker Hub (never downloading the base) and checks the layers are base + tested.
```bash
# notebook
tools/build_image.sh            # without TAG: only installs crane into /root/bin, then stops
export PATH=/root/bin:$PATH
crane auth login ghcr.io -u <github-user> --password-stdin   # classic GitHub token with write:packages
LAYERS_REPO=ghcr.io/<github-user>/ocr-layers TAG=v1 tools/build_image.sh
# laptop
gh workflow run assemble -f tag=v1 && gh run watch
```
The container starts `server.py`; the harness then runs `python3 /app/app.py --input-image ...`.

On a machine with Docker and ~60 GB of disk, `docker build -t ocr .` builds an equivalent image from the
`Dockerfile` (same layout and contents, different layer bytes).
