FROM python:3.11-slim

WORKDIR /app

# 1. Copy requirements first (better caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy FULL project INCLUDING artifacts
COPY . .

# 3. Force correct path safety
ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]