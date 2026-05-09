FROM python:3.11-slim

WORKDIR /app

# Copy everything
COPY . .

# Install exact versions from requirements
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]


