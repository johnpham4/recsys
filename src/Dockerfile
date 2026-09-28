# Use a slim version of the official Python image as a base image
FROM python:3.11.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential

# Copy only the requirements file first to leverage Docker layer caching
COPY src/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt && \
    apt-get remove -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

RUN rm -f /app/requirements.txt

# Copy application source code
COPY src /app/src

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
