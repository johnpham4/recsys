"""
Universal Modal GPU Runner for RecSys & Machine Learning.
Runs ANY training script, PyTorch Lightning module, or Jupyter notebook on serverless GPU
with shared volumes, automatic path symlinks, and strict cost timeouts.
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load local .env credentials before modal client initializes
load_dotenv()

import modal

# 1. Initialize Modal App
app = modal.App("recsys-universal-runner")

# 2. Shared Persistent Cloud Volumes
# /data: stores shared Parquet datasets, ID mappings, feature stores
# /checkpoints: stores trained model weights, MLflow/Evidently artifacts, checkpoints
dataset_vol = modal.Volume.from_name("recsys-dataset", create_if_missing=True)
checkpoint_vol = modal.Volume.from_name("recsys-checkpoints", create_if_missing=True)

# 3. Universal Container Image
# Syncs model/, src/, notebooks/, and scripts/ into /root
image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands("apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*")
    .pip_install(
        "torch>=2.2.0",
        "lightning>=2.2.0",
        "pandas>=2.0.0",
        "pyarrow>=14.0.0",
        "numpy<2.0",
        "loguru>=0.7.0",
        "pydantic>=2.0.0",
        "scikit-learn>=1.4.0",
        "evidently>=0.4.30",
        "papermill>=2.5.0",
        "ipykernel>=6.29.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
        "mlflow>=2.10.0",
    )
    .add_local_dir("model", remote_path="/root/model")

    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("scripts", remote_path="/root/scripts")
    .add_local_file(".env", remote_path="/root/.env")
    .add_local_file("notebooks/023-bert4rec.ipynb", remote_path="/root/notebooks/023-bert4rec.ipynb")
    .add_local_file("notebooks/021-sequence-model.ipynb", remote_path="/root/notebooks/021-sequence-model.ipynb")
    .add_local_file("notebooks/011_item2vec.ipynb", remote_path="/root/notebooks/011_item2vec.ipynb")
)

# 4. Remote Execution Engine with Cost Guardrails
@app.function(
    image=image,
    gpu="A10G",           # High-Performance: A10G (24GB VRAM, 31 TFLOPS) - ~4x faster than T4
    volumes={"/data": dataset_vol, "/checkpoints": checkpoint_vol},
    secrets=[modal.Secret.from_dotenv()],
    timeout=1800,         # 30 minutes hard timeout (max cost ~$0.30)
    retries=0             # No infinite retry loops
)

def execute_remote(cmd: str):
    """Executes a command on remote container with linked volumes and environment."""
    import os
    import sys
    import subprocess
    from pathlib import Path
    from dotenv import load_dotenv
    from loguru import logger

    if os.path.exists("/root/.env"):
        load_dotenv("/root/.env")

    logger.info("=" * 60)
    logger.info(f"🚀 REMOTE GPU LAUNCHER: {cmd}")
    logger.info("=" * 60)


    # 1. Setup path compatibility: ensure /root/data points to /data volume
    # and /root/output points to /checkpoints volume
    if not os.path.exists("/root/data"):
        try:
            os.symlink("/data", "/root/data")
            logger.info("Linked /data volume -> /root/data")
        except Exception as e:
            logger.warning(f"Could not symlink /data: {e}")

    if not os.path.exists("/root/checkpoints"):
        try:
            os.symlink("/checkpoints", "/root/checkpoints")
            logger.info("Linked /checkpoints volume -> /root/checkpoints")
        except Exception as e:
            logger.warning(f"Could not symlink /checkpoints: {e}")

    # Ensure src/ compatibility for legacy notebook imports (from src.bert4rec...)
    if not os.path.exists("/root/src/bert4rec"):
        try:
            os.symlink("/root/model/bert4rec", "/root/src/bert4rec")
        except Exception:
            pass
    if not os.path.exists("/root/src/id_mapper.py"):
        try:
            os.symlink("/root/model/id_mapper.py", "/root/src/id_mapper.py")
        except Exception:
            pass

    # 2. Setup environment variables
    env = os.environ.copy()
    env["PYTHONPATH"] = f"/root:{env.get('PYTHONPATH', '')}"
    env["DATA_DIR"] = "/data"
    env["CHECKPOINT_DIR"] = "/checkpoints"

    # 3. Run the user's command inside /root
    process = subprocess.Popen(
        cmd,
        shell=True,
        cwd="/root",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Stream output in real-time
    for line in iter(process.stdout.readline, ""):
        print(line, end="", flush=True)

    process.stdout.close()
    return_code = process.wait()

    # 4. Persist changes to checkpoints volume
    checkpoint_vol.commit()

    if return_code != 0:
        logger.error(f"❌ Command failed with return code: {return_code}")
        raise RuntimeError(f"Command execution failed: {cmd}")

    logger.info("✅ Command finished successfully. Checkpoints committed to Volume.")
    return return_code


# 5. Local Entrypoint: Upload Data to Volume (Run once)
@app.local_entrypoint()
def upload_data(local_dir: str = "data"):
    """
    Uploads files from local data directory to /data volume on Modal.
    Usage: uvx modal run scripts/modal_runner.py::upload_data
    """
    print("=" * 60)
    print(f"📤 Uploading local files from '{local_dir}' to Modal Volume 'recsys-dataset'...")
    print("=" * 60)

    data_path = Path(local_dir)
    if not data_path.exists():
        print(f"❌ Local directory '{local_dir}' not found!")
        return

    # Upload only essential dataset files required for training
    target_files = [
        "train_features_neg_df.parquet",
        "val_features_neg_df.parquet",
        "idm.json",
    ]

    with dataset_vol.batch_upload() as batch:
        for fname in target_files:
            local_fp = data_path / fname
            if local_fp.exists():
                size_mb = local_fp.stat().st_size / (1024 * 1024)
                print(f"  -> Uploading {fname} ({size_mb:.1f} MB)...")
                batch.put_file(str(local_fp), f"/{fname}")
            else:
                print(f"  ℹ️ File {fname} not found, skipping.")

    print("\n✅ Dataset upload completed! Files are permanently stored in Volume.")


# 6. Local Entrypoint: Download Checkpoints to Local
@app.local_entrypoint()
def download_checkpoints(remote_pattern: str = "/", local_target: str = "data"):
    """
    Downloads model checkpoints from Modal Volume to local directory.
    Usage: uvx modal run scripts/modal_runner.py::download_checkpoints
    """
    print(f"📥 Downloading files from Modal Volume 'recsys-checkpoints' to '{local_target}'...")
    import shutil
    modal_bin = shutil.which("modal") or shutil.which("uvx") or os.path.expanduser("~/.local/bin/uvx")
    if modal_bin.endswith("uvx"):
        cmd = [modal_bin, "modal", "volume", "get", "recsys-checkpoints", remote_pattern, local_target]
    else:
        cmd = [modal_bin, "volume", "get", "recsys-checkpoints", remote_pattern, local_target]

    
    env = os.environ.copy()
    local_bin = os.path.expanduser("~/.local/bin")
    if local_bin not in env.get("PATH", ""):
        env["PATH"] = f"{local_bin}:{env.get('PATH', '')}"
    subprocess.run(cmd, env=env)


# 7. Local Entrypoint: Universal Run Command
@app.local_entrypoint()
def run(cmd: str):
    """
    Executes any training script or command on Modal GPU.
    Usage examples:
      uvx modal run scripts/modal_runner.py::run --cmd "python notebooks/001-train-pipeline.py"
      uvx modal run scripts/modal_runner.py::run --cmd "papermill notebooks/023-bert4rec.ipynb output.ipynb"
    """
    print("=" * 60)
    print(f"🎯 LAUNCHING COMMAND ON MODAL GPU (T4, Serverless):")
    print(f"   $ {cmd}")
    print("🛡️ Guardrail: 30m timeout, auto scale-to-zero.")
    print("=" * 60)

    execute_remote.remote(cmd)
