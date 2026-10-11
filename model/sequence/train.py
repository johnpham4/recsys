"""
Standalone Training Entrypoint for Sequence Model (GRU4Rec / GRU Baseline).
Can be executed locally or inside Modal GPU container.
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import pandas as pd
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from loguru import logger

# Remove script directory from sys.path to prevent 'model.py' shadowing
script_dir = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p).resolve() != script_dir]

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.sequence.model import GRUModel, SequenceModel
from model.sequence.trainer import LitSequence
from model.dataset import UserItemBinaryDFDataset
from model.id_mapper import IDMapper


def parse_args():
    parser = argparse.ArgumentParser(description="Train GRU Sequential Model")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing parquet datasets")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--user-col", type=str, default="user_id", help="User ID column name")
    parser.add_argument("--item-col", type=str, default="parent_asin", help="Item ID column name")
    parser.add_argument("--rating-col", type=str, default="rating", help="Rating column name")
    parser.add_argument("--timestamp-col", type=str, default="timestamp", help="Timestamp column name")
    parser.add_argument("--top_k", type=int, default=10, help="Top-K for ranking")
    parser.add_argument("--top_K", type=int, default=100, help="Top-K for recall evaluation")
    parser.add_argument("--eval-ranking", action="store_true", default=True, help="Evaluate ranking metrics at end")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("=" * 60)
    logger.info(f"🚀 Initializing GRU Sequence Training Pipeline (Epochs: {args.epochs}, Batch: {args.batch_size})")
    logger.info("=" * 60)

    # 1. Resolve paths
    data_dir = Path("/data") if Path("/data/train_features_neg_df.parquet").exists() else Path(args.data_dir)
    ckpt_dir = Path("/checkpoints") if Path("/checkpoints").exists() else Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train_features_neg_df.parquet"
    val_path = data_dir / "val_features_neg_df.parquet"
    idm_path = data_dir / "idm.json"

    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found at {train_path}!")

    # 2. Load Parquet Data
    logger.info(f"Loading datasets from {data_dir}...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    idm = IDMapper().load(str(idm_path)) if idm_path.exists() else None

    # Determine num_users & num_items
    n_users = int(max(train_df["user_indice"].max(), val_df["user_indice"].max()) + 1)
    n_items = int(max(train_df["item_indice"].max(), val_df["item_indice"].max()) + 1)
    logger.info(f"Vocabulary sizes: num_users={n_users:,} | num_items={n_items:,} | train_samples={len(train_df):,}")

    # Binarize ratings for BCELoss: rating > 0 is 1.0 (positive interaction), otherwise 0.0 (negative)
    train_df[args.rating_col] = (train_df[args.rating_col] > 0).astype("float32")
    val_df[args.rating_col] = (val_df[args.rating_col] > 0).astype("float32")

    # 3. Create Datasets & DataLoaders
    train_dataset = UserItemBinaryDFDataset(
        df=train_df,
        user_col="user_indice",
        item_col="item_indice",
        rating_col=args.rating_col,
        timestamp_col=args.timestamp_col,
    )
    val_dataset = UserItemBinaryDFDataset(
        df=val_df,
        user_col="user_indice",
        item_col="item_indice",
        rating_col=args.rating_col,
        timestamp_col=args.timestamp_col,
    )

    num_workers = 4 if torch.cuda.is_available() else 0
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # 4. Instantiate GRU Model
    model = GRUModel(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=args.embed_dim,
        dropout=args.dropout,
    )

    # 5. Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="gru-sequence-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
        verbose=True,
    )

    # 6. Lightning Module & MLflow Setup
    from dotenv import load_dotenv
    load_dotenv()

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    mlf_logger = None
    if mlflow_uri:
        from lightning.pytorch.loggers import MLFlowLogger
        mlf_logger = MLFlowLogger(
            experiment_name="recsys",
            run_name="002-sequence",
            tracking_uri=mlflow_uri,
            log_model=True,
        )
        logger.info(f"🔗 Connected MLflow tracking: {mlflow_uri}")

    lit_model = LitSequence(
        model=model,
        learning_rate=args.lr,
        checkpoint_callback=checkpoint_callback,
        evaluate_ranking=args.eval_ranking and (idm is not None),
        idm=idm,
        args=args,
        log_dir=str(ckpt_dir),
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
    )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stopping],
        logger=mlf_logger,
        enable_progress_bar=True,
        log_every_n_steps=20,
    )

    # 7. Start Training
    logger.info("⚡ Training GRU model started...")
    trainer.fit(lit_model, train_loader, val_loader)

    best_ckpt = checkpoint_callback.best_model_path
    logger.info("=" * 60)
    logger.info(f"🎉 GRU Training finished successfully!")
    logger.info(f"🏆 Best model checkpoint saved to: {best_ckpt}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
