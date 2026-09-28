import os
import torch
import lightning as l
import torch.nn as nn
from torchmetrics import AUROC
import pandas as pd
from loguru import logger
from pydantic import BaseModel
from sklearn.metrics import roc_auc_score

from evidently import Report, Dataset, DataDefinition, Recsys, BinaryClassification
from evidently.presets import ClassificationPreset
from evidently.metrics import (
    FBetaTopK,
    NDCG,
    Personalization,
    PrecisionTopK,
    RecallTopK,
)

try:
    from model.eval.utils import create_label_df, create_rec_df, merge_recs_with_target
    from model.id_mapper import IDMapper
except ImportError:
    from src.eval.utils import create_label_df, create_rec_df, merge_recs_with_target
    from src.id_mapper import IDMapper
from .model import SequenceModel


class LitSequence(l.LightningModule):
    def __init__(
        self,
        model: SequenceModel,
        learning_rate: float = 0.001,
        l2_reg: float = 1e-5,
        log_dir: str = ".",
        evaluate_ranking: bool = False,
        idm: IDMapper = None,
        args: BaseModel = None,
        checkpoint_callback=None,
        accelerator: str = "cpu",
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
        self.val_roc_auc_metric = AUROC(task="binary")

        self.save_hyperparameters(
            {
                "n_users": self.model.n_users,
                "n_items": self.model.n_items,
                "embedding_dim": self.model.item_embedding.embedding_dim,
                "item_embedding": self.model.item_embedding,
                "dropout": self.model.dropout
            }
        )

    def training_step(self, batch, batch_idx):
        user_ids = batch["user"]
        item_sequences = batch["item_sequence"]
        target_items = batch["item"]
        labels = batch["rating"]
        loss_fn = self._get_loss_fn()
        predictions = self.model(user_ids, target_items, item_sequences).view(labels.shape)
        loss = loss_fn(predictions, labels)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        user_ids = batch["user"]
        item_sequences = batch["item_sequence"]
        target_items = batch["item"]
        labels = batch["rating"]
        loss_fn = self._get_loss_fn()

        predictions = self.model(user_ids, target_items, item_sequences).view(labels.shape)
        loss = loss_fn(predictions, labels)

        self.val_roc_auc_metric.update(predictions, labels.int())
        current_roc_auc = self.val_roc_auc_metric.compute()

        self.log("val_roc_auc", current_roc_auc, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=2),
            "monitor": "val_loss",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_validation_epoch_end(self):
        sch = self.lr_schedulers()
        if sch is not None:
            self.log("learning_rate", sch.get_last_lr()[0], sync_dist=True)
        roc_auc = self.val_roc_auc_metric.compute()
        self.log("val_roc_auc", roc_auc, sync_dist=True)
        self.val_roc_auc_metric.reset()

    def on_fit_end(self):
        if self.checkpoint_callback:
            logger.info(f"Loading best model from {self.checkpoint_callback.best_model_path}...")
            self.model = LitSequence.load_from_checkpoint(
                self.checkpoint_callback.best_model_path, model=self.model
            ).model

        self.model = self.model.to(self._get_device())
        logger.info("Logging classification metrics...")
        self._log_classification_metrics()
        if self.evaluate_ranking:
            logger.info("Logging ranking metrics...")
            self._log_ranking_metrics()

    def _log_classification_metrics(self):
        self.model.eval()
        val_loaders = self.trainer.val_dataloaders
        if isinstance(val_loaders, list):
            val_loader = val_loaders[0]
        else:
            val_loader = val_loaders

        labels, classifications = [], []
        for _, batch in enumerate(val_loader):
            users = batch["user"].to(self._get_device())
            items = batch["item"].to(self._get_device())
            seqs = batch["item_sequence"].to(self._get_device())
            y_true = batch["rating"].to(self._get_device())
            y_pred = self.model.predict(users, seqs, items).view(y_true.shape)
            labels.extend(y_true.cpu().detach().numpy())
            classifications.extend(y_pred.cpu().detach().numpy())

        eval_df = pd.DataFrame({"labels": labels, "classification_prob": classifications})
        eval_df["label"] = eval_df["labels"].gt(0).astype(int)
        self.eval_classification_df = eval_df

        data_def = DataDefinition(
            numerical_columns=["classification_prob"],
            categorical_columns=["label"],
            classification=[BinaryClassification(
                target="label",
                prediction_labels="classification_prob"
            )]
        )
        eval_data = Dataset.from_pandas(
            eval_df[["label", "classification_prob"]],
            data_definition=data_def
        )
        report = Report(metrics=[ClassificationPreset()])
        snapshot = report.run(current_data=eval_data, reference_data=None)
        report_path = f"{self.log_dir}/evidently_report_classification.html"
        os.makedirs(self.log_dir, exist_ok=True)
        snapshot.save_html(report_path)

        if "mlflow" in str(self.logger.__class__).lower():
            run_id = self.logger.run_id
            client = self.logger.experiment
            client.log_artifact(run_id, report_path)

            # Calculate ROC-AUC using sklearn (Evidently 0.7.15 has different structure)
            y_true = eval_df["label"].values
            y_score = eval_df["classification_prob"].values
            roc_auc = roc_auc_score(y_true, y_score)
            client.log_metric(run_id, "val_roc_auc", float(roc_auc))

    def _log_ranking_metrics(self):
        self.model.eval()
        args, idm = self.args, self.idm

        val_df = self.trainer.val_dataloaders.dataset.df
        to_rec_df = val_df.sort_values(args.timestamp_col, ascending=True).drop_duplicates(subset=[args.user_col])
        label_df = create_label_df(val_df, args.user_col, args.item_col, args.rating_col, args.timestamp_col)
        with torch.no_grad():
            recs = self.model.recommend(
                torch.tensor(to_rec_df["user_indice"].values, device=self._get_device()),
                torch.tensor(to_rec_df["item_sequence"].values.tolist(), device=self._get_device()),
                k=args.top_K,
                batch_size=4,
            )
        rec_df = pd.DataFrame(recs).pipe(create_rec_df, idm, args.user_col, args.item_col)
        eval_df = merge_recs_with_target(rec_df, label_df, k=args.top_K,
                                         user_col=args.user_col, item_col=args.item_col, rating_col=args.rating_col)
        self.eval_ranking_df = eval_df

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
                NDCG(k=args.top_k),
                RecallTopK(k=args.top_K),
                PrecisionTopK(k=args.top_k),
                FBetaTopK(k=args.top_k),
                Personalization(k=args.top_k),
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

            # Access metrics from snapshot.metric_results in Evidently 0.7.15
            for metric_id, metric_result in snapshot.metric_results.items():
                metric_name = metric_result.explicit_metric_id() if hasattr(metric_result, 'explicit_metric_id') else ""

                if "PersonalizationMetric" in metric_name:
                    if hasattr(metric_result, 'value'):
                        client.log_metric(run_id, f"val_PersonalizationMetric", float(metric_result.value))
                    continue

                # For metrics with k values (NDCG, Recall, Precision, FBeta)
                if any(x in metric_name for x in ["NDCG", "Recall", "Precision", "FBeta"]):
                    if hasattr(metric_result, 'value') and isinstance(metric_result.value, dict):
                        for k, v in metric_result.value.items():
                            client.log_metric(run_id, f"val_{metric_name}_at_k_as_step", float(v), step=int(k))

    def _get_loss_fn(self):
        return nn.BCELoss()

    def _get_device(self):
        return self.accelerator
