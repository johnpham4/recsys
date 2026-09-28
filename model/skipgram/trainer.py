import os

import lightning as L
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, precision_recall_curve

from evidently import Report, Dataset, DataDefinition, BinaryClassification
from evidently.presets import ClassificationPreset

from .model import SkipGram


class LitSkipGram(L.LightningModule):
    def __init__(
        self,
        skipgram_model: SkipGram,
        learning_rate: float = 0.001,
        l2_reg: float = 1e-5,
        log_dir: str = ".",
    ):
        super().__init__()
        self.skipgram_model = skipgram_model
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.log_dir = log_dir

        # Save hyperparameters for inference
        self.save_hyperparameters({
            "num_items": skipgram_model.embeddings.num_embeddings - 1,
            "embedding_dim": skipgram_model.embeddings.embedding_dim,
        })

    def training_step(self, batch, batch_idx):
        target_items = batch["target_items"]
        context_items = batch["context_items"]

        predictions = self.skipgram_model.forward(target_items, context_items)
        labels = batch["labels"].float().squeeze()

        loss_fn = nn.BCELoss()
        loss = loss_fn(predictions, labels)

        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        target_items = batch["target_items"]
        context_items = batch["context_items"]

        predictions = self.skipgram_model.forward(target_items, context_items)
        labels = batch["labels"].float().squeeze()

        loss_fn = nn.BCELoss()
        loss = loss_fn(predictions, labels)

        self.log(
            "val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        # Create the optimizer
        optimizer = torch.optim.Adam(
            self.skipgram_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_reg,
        )

        # Create the scheduler
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.3, patience=2
            ),
            "monitor": "val_loss",
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_validation_epoch_end(self):
        sch = self.lr_schedulers()

        if sch is not None:
            self.log("learning_rate", sch.get_last_lr()[0], sync_dist=True)

    def on_fit_end(self):
        self._log_classification_metrics(self.trainer.val_dataloaders)

    def _log_classification_metrics(self, val_loader):
        target_items = []
        context_items = []
        labels = []

        for _, batch_input in enumerate(val_loader):
            _target_items = batch_input["target_items"].cpu().detach().numpy()
            _context_items = batch_input["context_items"].cpu().detach().numpy()
            _labels = batch_input["labels"].cpu().detach().numpy()

            target_items.extend(_target_items)
            context_items.extend(_context_items)
            labels.extend(_labels)

        val_df = pd.DataFrame(
            {
                "target_items": target_items,
                "context_items": context_items,
                "labels": labels,
            }
        )

        target_items = torch.tensor(val_df["target_items"].values, device=self.device)
        context_items = torch.tensor(val_df["context_items"].values, device=self.device)
        classifications = self.skipgram_model(target_items, context_items)

        eval_classification_df = val_df.assign(
            classification_proba=classifications.cpu().detach().numpy(),
            label=lambda df: df["labels"].astype(int),
        )

        # Evidently
        target_col = "label"
        prediction_col = "classification_proba"

        # DataDefinition replaces ColumnMapping in Evidently 0.7.x
        data_def = DataDefinition(
            numerical_columns=[prediction_col],
            categorical_columns=[target_col],
            classification=[BinaryClassification(
                target=target_col,
                prediction_labels=prediction_col
            )]
        )

        current_dataset = Dataset.from_pandas(
            eval_classification_df[[target_col, prediction_col]],
            data_definition=data_def,
        )

        classification_performance_report = Report(
            metrics=[
                ClassificationPreset(),
            ]
        )

        snapshot = classification_performance_report.run(
            reference_data=None,
            current_data=current_dataset,
        )

        evidently_report_fp = f"{self.log_dir}/evidently_report_classification.html"
        os.makedirs(self.log_dir, exist_ok=True)
        snapshot.save_html(evidently_report_fp)

        if "mlflow" in str(self.logger.__class__).lower():
            run_id = self.logger.run_id
            mlf_client = self.logger.experiment
            mlf_client.log_artifact(run_id, evidently_report_fp)

            # Calculate ROC-AUC using sklearn (Evidently 0.7.15 has different structure)
            y_true = eval_classification_df[target_col].values
            y_score = eval_classification_df[prediction_col].values
            roc_auc = roc_auc_score(y_true, y_score)
            mlf_client.log_metric(run_id, "val_roc_auc", float(roc_auc))

            # Calculate Precision-Recall curve at probability thresholds
            precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)

            # Log at probability thresholds matching Evidently 0.6.5 ClassificationPRTable format
            # Log at 0%, 10%, 20%, ..., 100%
            for prob_pct in range(0, 101, 10):
                prob_threshold = prob_pct / 100.0

                # Find closest threshold in PR curve
                if len(thresholds) == 0:
                    continue

                if prob_threshold <= thresholds.min():
                    idx = 0
                elif prob_threshold >= thresholds.max():
                    idx = len(thresholds) - 1
                else:
                    idx = (abs(thresholds - prob_threshold)).argmin()

                precision = float(precisions[idx])
                recall = float(recalls[idx])

                mlf_client.log_metric(
                    run_id,
                    "val_precision_at_prob_as_threshold_step",
                    precision,
                    step=prob_pct,
                )
                mlf_client.log_metric(
                    run_id,
                    "val_recall_at_prob_as_threshold_step",
                    recall,
                    step=prob_pct,
                )
