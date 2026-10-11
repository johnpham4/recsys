.PHONY:
.ONESHELL:

PATH := $(HOME)/.local/bin:$(PATH)
export PATH

include .env
export

mlflow-up:
	docker compose -f compose.yml up -d mlflow_db mlflow_server minio minio_init

ml-platform-up:
	docker compose -f compose.yml up -d mlflow_server redis qdrant dwh

ml-platform-logs:
	- docker compose -f compose.yml logs -f

airflow-up:
	docker compose -f compose.airflow.yml up -d

airflow-logs:
	- docker compose -f compose.airflow.yml logs -f

lab:
	uv run jupyter lab --port 8888 --host 0.0.0.0

ui-up:
	docker compose -f compose.frontend.yml up -d

ui-logs:
	docker compose -f compose.frontend.yml logs -f

api-up:
	docker compose -f compose.api.yml up -d

api-logs:
	docker compose -f compose.api.yml logs -f

api-down:
	docker compose -f compose.api.yml down

# Create the requirements.txt file and update the torch to CPU version to reduce the image size
requirements-txt:
	uv export --group serving --group ml --no-hashes --format requirements-txt > requirements.txt
	# Commend out torch in requirements.txt to pre-install the CPU version in Docker
	sed '/^torch/ s/^/# /' requirements.txt > .tmp && mv .tmp requirements.txt
	sed '/^nvidia/ s/^/# /' requirements.txt > .tmp && mv .tmp requirements.txt

build-pipeline:
	docker build -f feature_pipeline.Dockerfile . -t recsys-mvp-pipeline:0.0.1

feature-server-up:
	docker compose -f compose.yml up -d feature_online_server feature_offline_server feature_store_ui

down:
	docker compose -f compose.yml down
	docker compose -f compose.airflow.yml down
	docker compose -f compose.pipeline.yml down
	docker compose -f compose.api.yml down

remove-feature-store-data:
	rm -rf data/redis
	rm -rf data/postgres

remove-data: remove-feature-store-data
	rm -rf data/mlflow
	rm -rf data/qdrant_storage

clean: down remove-data

clean-feature-store: down remove-feature-store-data

modal-status:
	uv run python modal/modal_track.py

modal-upload:
	uv run --with modal modal run modal/modal_runner.py::upload_data

modal-train-bert:
	uv run --with modal modal run modal/modal_runner.py::run --cmd "python model/bert4rec/train.py --epochs 8 --batch-size 128"

modal-train-gru:
	uv run --with modal modal run modal/modal_runner.py::run --cmd "python model/sequence/train.py --epochs 8 --batch-size 512"

modal-train: modal-train-bert

modal-benchmark:
	uv run --with modal modal run modal/modal_runner.py::run --cmd "python scripts/benchmark_serving.py"


modal-download:
	uv run --with modal modal run modal/modal_runner.py::download_checkpoints

modal-delete-dataset:
	uv run --with modal modal volume delete -y recsys-dataset

