# Use a slim version of the official Python image as a base image
FROM python:3.11.9-slim

WORKDIR /app

# Install Torch first because it's required but slow to install
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu && \
    apt-get remove -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt /app/

RUN pip install -r requirements.txt && \
    apt-get remove -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Remove as no need after installation
RUN rm -f requirements.txt

COPY src /app/src
COPY model /app/model

ENV PYTHONPATH=/app

EXPOSE 3000

CMD ["bentoml", "serve", "src.infra.model_server.service:SeqRPService"]
