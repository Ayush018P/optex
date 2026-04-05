FROM python:3.11-slim AS base
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python scripts/generate_sample_data.py
RUN python scripts/pretrain_agents.py || true

# Runtime image
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=base /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=base /usr/local/bin /usr/local/bin
COPY --from=base /app /app

RUN useradd -m optex
USER optex
# Hugging Face Spaces requires port 7860
ENV PORT=7860
EXPOSE 7860
CMD ["gunicorn", "dashboard.app:server", "-k", "gevent", "-b", "0.0.0.0:7860", "--timeout", "120"]

#http://localhost:8050/live
