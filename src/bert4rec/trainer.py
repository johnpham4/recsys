import os
import torch
import lightning as L
import torch.nn as nn
import pandas as pd
from loguru import logger
from pydantic import BaseModel
import numpy as np

from evidently import Report, Dataset, DataDefinition, Recsys
from evidently.metrics import (
    NDCG,
    RecallTopK,
    PrecisionTopK,
)

from src.eval.utils import create_label_df, create_rec_df, merge_recs_with_target
from src.id_mapper import IDMapper
from .model import BERT4Rec
from .dataset import BERT4RecDataset


class LitBERT4Rec(L.LightningModule):
    """Lightning trainer for BERT4Rec model."""

    def __init__(
        self,
        model: BERT4Rec,
        learning_rate: float = 0.001,
        l2_reg: float = 1e-5,
        log_dir: str = ".",
        evaluate_ranking: bool = True,  # Always evaluate ranking for BERT4Rec
        idm: IDMapper = None,
        args: BaseModel = None,
        checkpoint_callback=None,
        accelerator: str = "cpu",
        mask_prob: float = 0.15,
        top_k: int = 10  # Top-K for ranking metrics
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.log_dir = log_dir
        self.evaluate_ranking = evaluate_ranking
        self.idm = idm
        self.args = args
        self.accelerator = accelerator
        self.checkpoint_callback = checkpoint_callback
        self.mask_prob = mask_prob
        self.top_k = top_k

        self.save_hyperparameters(
            {
                "num_items": self.model.num_items,
                "max_seq_len": self.model.max_seq_len,
                "embed_dim": self.model.embed_dim,
                "num_layers": len(self.model.transformer_blocks),
                "num_heads": self.model.transformer_blocks[0].attention.num_heads,
                "ff_dim": self.model.transformer_blocks[0].feed_forward.fc1.out_features,
                "dropout_prob": self.model.dropout.p,
                "mask_prob": mask_prob,
                "top_k": top_k
            }
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        padding_idx_tensor = torch.tensor(self.model.padding_idx, device=input_ids.device)
        labels = torch.where(labels == -1, padding_idx_tensor, labels)

        # Forward pass
        logits = self.model(input_ids, attention_mask)

        # Calculate loss (only for masked positions)
        # Create mask for non-padding positions
        valid_mask = (labels != self.model.padding_idx).float()

        # Cross-entropy loss
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.model.padding_idx, reduction='none')
        loss_per_position = loss_fn(
            logits.view(-1, self.model.num_items + 1),
            labels.view(-1)
        )
        loss_per_position = loss_per_position.view(input_ids.size())

        # Apply valid mask and average
        masked_loss = (loss_per_position * valid_mask).sum() / (valid_mask.sum() + 1e-9)

        self.log("train_loss", masked_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return masked_loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        # Map -1 padding to padding_idx (like model forward)
        padding_idx_tensor = torch.tensor(self.model.padding_idx, device=input_ids.device)
        labels = torch.where(labels == -1, padding_idx_tensor, labels)

        # Forward pass
        logits = self.model(input_ids, attention_mask)

        # Calculate loss (masked language model loss)
        valid_mask = (labels != self.model.padding_idx).float()
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.model.padding_idx, reduction='none')
        loss_per_position = loss_fn(
            logits.view(-1, self.model.num_items + 1),
            labels.view(-1)
        )
        loss_per_position = loss_per_position.view(input_ids.size())

        masked_loss = (loss_per_position * valid_mask).sum() / (valid_mask.sum() + 1e-9)

        self.log("val_loss", masked_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return masked_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_reg
        )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.3,
                patience=2
            ),
            "monitor": "val_loss",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_validation_epoch_end(self):
        sch = self.lr_schedulers()
        if sch is not None:
            self.log("learning_rate", sch.get_last_lr()[0], sync_dist=True)

    def on_fit_end(self):
        if self.checkpoint_callback:
            logger.info(f"Loading best model from {self.checkpoint_callback.best_model_path}...")
            self.model = LitBERT4Rec.load_from_checkpoint(
                self.checkpoint_callback.best_model_path,
                model=self.model
            ).model

        self.model = self.model.to(self._get_device())

        if self.evaluate_ranking:
            logger.info("Logging ranking metrics...")
            self._log_ranking_metrics()

    def _log_ranking_metrics(self):
        """Log ranking metrics for recommendation quality (NDCG@K, HR@K, Recall@K)."""
        self.model.eval()
        args, idm = self.args, self.idm

        val_df = self.trainer.val_dataloaders.dataset.df
        to_rec_df = val_df.sort_values(args.timestamp_col, ascending=True).drop_duplicates(subset=[args.user_col])
        label_df = create_label_df(val_df, args.user_col, args.item_col, args.rating_col, args.timestamp_col)

        inference_dataset = BERT4RecDataset(
            df=to_rec_df,
            max_seq_len=self.model.max_seq_len,
            padding_idx=self.model.padding_idx,  # Use model's padding_idx (n_items)
            for_inference=True
        )

        inference_loader = torch.utils.data.DataLoader(
            inference_dataset,
            batch_size=32,
            shuffle=False
        )

        # Generate recommendations (like sequence model: 1 row per user)
        all_user_indices = []
        all_recs = []
        all_scores = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(inference_loader):
                input_ids = batch["input_ids"].to(self._get_device())
                attention_mask = batch["attention_mask"].to(self._get_device())

                top_k_items, top_k_scores = self.model.recommend(
                    input_ids,
                    k=self.top_k,
                    attention_mask=attention_mask
                )

                batch_size = top_k_items.size(0)
                for i in range(batch_size):
                    user_idx = batch_idx * 32 + i
                    if user_idx < len(to_rec_df):
                        user_indice = to_rec_df["user_indice"].values[user_idx]
                        all_user_indices.extend([user_indice] * self.top_k)
                        all_recs.extend(top_k_items[i].cpu().flatten().tolist())
                        all_scores.extend(top_k_scores[i].cpu().flatten().tolist())

        rec_df = pd.DataFrame({
            "user_indice": all_user_indices,
            "recommendation": all_recs,
            "score": all_scores
        })

        logger.info(f"rec_df shape: {rec_df.shape}")
        logger.info(f"rec_df dtypes: {rec_df.dtypes}")
        logger.info(f"rec_df sample: {rec_df.head()}")
        logger.info(f"First recommendation value: {rec_df['recommendation'].iloc[0]}")

        rec_df = create_rec_df(rec_df, idm, args.user_col, args.item_col)
        eval_df = merge_recs_with_target(
            rec_df, label_df, k=self.top_k,
            user_col=args.user_col, item_col=args.item_col, rating_col=args.rating_col
        )

        # Create Evidently report for ranking metrics (like sequence model)
        data_def = DataDefinition(
            ranking=[Recsys(
                user_id=args.user_col,
                item_id=args.item_col,
                target=args.rating_col,
                prediction="rec_ranking",
            )]
        )

        current_data = Dataset.from_pandas(eval_df, data_definition=data_def)
        report = Report(
            metrics=[
                NDCG(k=self.top_k),
                RecallTopK(k=self.top_k),
                PrecisionTopK(k=self.top_k),
            ]
        )

        snapshot = report.run(current_data=current_data, reference_data=None)
        report_path = f"{self.log_dir}/evidently_report_ranking.html"
        os.makedirs(self.log_dir, exist_ok=True)
        snapshot.save_html(report_path)

        if "mlflow" in str(self.logger.__class__).lower():
            run_id = self.logger.run_id
            client = self.logger.experiment
            client.log_artifact(run_id, report_path)


    def _get_device(self):
        return self.accelerator
