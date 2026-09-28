import numpy as np
import mlflow
import mlflow.pyfunc
import torch
import torch.nn as nn

try:
    from model.id_mapper import IDMapper
except ImportError:
    from src.id_mapper import IDMapper
from .model import SequenceModel

class SequenceModelWrapper(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        model_path = context.artifacts["model_path"]
        checkpoint = torch.load(model_path, map_location="cpu")

        # hyperparams
        n_users = checkpoint["hyper_parameters"]["n_users"]
        n_items = checkpoint["hyper_parameters"]["n_items"]
        embedding_dim = checkpoint["hyper_parameters"]["embedding_dim"]
        item_embedding = checkpoint["hyper_parameters"]["item_embedding"]

        # Handle dropout parameter - it might be a Dropout object or float
        dropout_param = checkpoint["hyper_parameters"]["dropout"]
        if hasattr(dropout_param, 'p'):  # It's a Dropout object
            dropout = dropout_param.p
        else:  # It's already a float
            dropout = dropout_param

        print(f"DEBUG: dropout type: {type(dropout)}, value: {dropout}")

        state_dict = checkpoint["state_dict"]

        # Remove 'model.' prefix from keys if it exists (Lightning saves with this prefix)
        clean_state_dict = {
            k.replace("model.", ""): v for k, v in state_dict.items()
        }

        base_model = SequenceModel(n_users, n_items, embedding_dim, item_embedding, dropout)
        base_model.load_state_dict(clean_state_dict, strict=True)

        self.model = base_model
        self.model.eval()

        json_path = context.artifacts["id_mapping"]
        self.idm = IDMapper().load(json_path)

    def predict(self, context, model_input, params=None):
        sequence_length = 10
        padding_value = -1

        if not isinstance(model_input, dict):
            # Ref: https://github.com/mlflow/mlflow/issues/11930
            model_input = model_input.to_dict(orient="records")[0]
        user_indices = [self.idm.get_user_index(id_) for id_ in model_input["user_ids"]]
        item_indices = [self.idm.get_item_index(id_) for id_ in model_input["item_ids"]]
        item_sequences = []
        for item_sequence in model_input["item_sequences"]:
            item_sequence = [self.idm.get_item_index(id_) for id_ in item_sequence]
            padding_needed = sequence_length - len(item_sequence)
            item_sequence = np.pad(
                item_sequence,
                (padding_needed, 0),
                "constant",
                constant_values=padding_value,
            )
            item_sequences.append(item_sequence)
        infer_output = self.infer(user_indices, item_sequences, item_indices).tolist()
        return {
            **model_input,
            "scores": infer_output,
        }


    def infer(self, user_indices, item_sequences, item_indices):
        user_indices = torch.tensor(user_indices)
        item_sequences = torch.tensor(item_sequences)
        item_indices = torch.tensor(item_indices)
        output = self.model.predict(user_indices, item_sequences, item_indices)
        return output.view(len(user_indices)).detach().numpy()