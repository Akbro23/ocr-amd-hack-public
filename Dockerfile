# syntax=docker/dockerfile:1
# Reference build for machines that have Docker (and ~60 GB of disk). It produces the same layout as the
# submitted image, which is built WITHOUT Docker (tools/build_image.sh on the notebook + tools/assemble.sh
# in CI, see README): packages in /opt/ocr/deps on PYTHONPATH, weights in /models, code in /app.
# Same contents, not byte-identical layers. Keep it in sync with build_image.sh and image_config.py.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

# torch comes from the base image: drop it (and everything only it needs) from the resolution
COPY pyproject.toml /tmp/pyproject.toml
RUN pip install uv \
 && echo 'torch ; sys_platform == "never"' > /tmp/no-torch.txt \
 && uv pip compile /tmp/pyproject.toml --python /opt/venv/bin/python --overrides /tmp/no-torch.txt -o /tmp/reqs.txt \
 && uv pip install --python /opt/venv/bin/python --no-deps --target /opt/ocr/deps -r /tmp/reqs.txt
ENV PYTHONPATH=/opt/ocr/deps

# weights baked in (not downloaded at startup), pinned revision
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-9B', \
      revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a', local_dir='/models/Qwen3.5-9B')" \
 && rm -rf /models/Qwen3.5-9B/.cache
ENV OCR_MODEL=/models/Qwen3.5-9B

COPY app/ /app/
RUN mkdir -p /app/input /app/output

# the socket exists only once the model is loaded and warm
HEALTHCHECK --interval=5s --start-period=600s CMD test -S /tmp/ocr.sock
ENTRYPOINT []
CMD ["python3", "/app/server.py"]
