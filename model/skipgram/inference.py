import json

import mlflow
import torch


import json
import mlflow
import torch
from .model import SkipGram
from .trainer import LitSkipGram


class SkipGramInferenceWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """
        Load model và id_mapping khi MLflow load lại model.
        """
        model_path = context.artifacts["model_path"]
        checkpoint = torch.load(model_path, map_location="cpu")

        n_items = checkpoint["hyper_parameters"]["num_items"]
        embedding_dim = checkpoint["hyper_parameters"]["embedding_dim"]

        base_model = SkipGram(n_items, embedding_dim)
        state_dict = checkpoint["state_dict"]

        clean_state_dict = {
            k.replace("skipgram_model.", ""): v for k, v in state_dict.items()
        }

        base_model.load_state_dict(clean_state_dict, strict=True)


        self.model = base_model
        self.model.eval()

        json_path = context.artifacts["id_mapping"]
        with open(json_path, "r") as f:
            self.id_mapping = json.load(f)


    def predict(self, context, model_input, params=None):
        """
        Args:
            context: The context object in mlflow.pyfunc often contains pointers to artifacts that are logged alongside the model during training (like feature encoders, embeddings, etc.)
        """
        if not isinstance(model_input, dict):
            # This is to work around the issue where MLflow automatically convert dict input into Dataframe
            # Ref: https://github.com/mlflow/mlflow/issues/11930
            model_input = model_input.to_dict(orient="records")[0]
        item_1_indices = [
            self.id_mapping["id_to_idx"].get(id_) for id_ in model_input["item_1_ids"]
        ]
        item_2_indices = [
            self.id_mapping["id_to_idx"].get(id_) for id_ in model_input["item_2_ids"]
        ]
        infer_output = self.infer(item_1_indices, item_2_indices).tolist()
        return {
            "item_1_ids": model_input["item_1_ids"],
            "item_2_ids": model_input["item_2_ids"],
            "scores": infer_output,
        }

    def infer(self, item_1_indices, item_2_indices):
        item_1_indices = torch.tensor(item_1_indices)
        item_2_indices = torch.tensor(item_2_indices)
        output = self.model(item_1_indices, item_2_indices)
        return output.detach().numpy()
