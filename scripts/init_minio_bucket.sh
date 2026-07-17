#!/bin/bash
# Create MinIO bucket for MLflow artifacts

set -e

MINIO_USER=${MINIO_ROOT_USER:-minio}
MINIO_PASS=${MINIO_ROOT_PASSWORD:-minio123}
BUCKET_NAME=${AWS_BUCKET_NAME:-data-recsys}

export MC_HOST_minio="http://${MINIO_USER}:${MINIO_PASS}@localhost:9000"

echo "Waiting for MinIO to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:9000/minio/health/live > /dev/null; then
        echo "MinIO is ready"
        break
    fi
    sleep 2
done

echo "Creating bucket '${BUCKET_NAME}'..."
mc mb minio/${BUCKET_NAME} || echo "Bucket may already exist"

echo "Done"
