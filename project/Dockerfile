FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./requirements.txt
# The official CPU wheel satisfies the locked public version (2.15.0) without
# pulling several gigabytes of CUDA libraries into this CPU-only smoke image.
RUN python -m pip install --no-cache-dir --only-binary=:all: --no-deps \
      torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir --only-binary=:all: -r requirements.txt

COPY . .
CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]
