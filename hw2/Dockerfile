FROM python:3.12-slim-bookworm

WORKDIR /work
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY solution.py ranking.py requirements.txt /work/
COPY scripts /work/scripts
COPY tests /work/tests
CMD ["python", "-m", "unittest", "discover", "-s", "tests", "-v"]
