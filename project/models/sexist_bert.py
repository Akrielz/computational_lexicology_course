from typing import Optional, Callable, Literal, List, Union, Tuple

import torch
from torch import nn
from tqdm import tqdm
from vision_models_playground.components.attention import TransformerEncoder, FeedForward

from project.models.average_reducer import AverageReducer
from project.models.bert_pooler import BertPooler
from project.pipeline.data_loader import DataLoader
from project.task_a.toxic_bert_encoder import ToxicBertEncoder


class SexistBert(nn.Module):
    def __init__(
            self,
            detoxify_model: Optional[str] = "original",
            depth: int = 2,
            apply_rotary_emb: bool = True,
            activation: Optional[Callable] = None,
            drop_path: float = 0.0,
            norm_type: Literal['pre_norm', 'post_norm'] = "pre_norm",
            num_classes: int = 2,
            device: str = "cpu",
            pool_method: Literal['average', 'bert'] = "average",
    ):
        super(SexistBert, self).__init__()

        detoxify_model_name = detoxify_model
        if detoxify_model is None:
            detoxify_model_name = "original"

        self.toxic_bert_encoder = ToxicBertEncoder(detoxify_model_name, device)

        config = self.toxic_bert_encoder.bert_model.config

        if detoxify_model is None:
            del self.toxic_bert_encoder
            self.toxic_bert_encoder = None

        dim = config.hidden_size
        heads = config.num_attention_heads
        dim_head = dim // heads
        mlp_dim = config.intermediate_size
        mlp_dropout = config.hidden_dropout_prob
        attention_dropout = config.attention_probs_dropout_prob

        if activation is None:
            activation = nn.ReLU()

        self.transformer = TransformerEncoder(
            dim=dim,
            depth=depth,
            heads=heads,
            head_dim=dim_head,
            mlp_dim=mlp_dim,
            mlp_dropout=mlp_dropout,
            attention_dropout=attention_dropout,
            apply_rotary_emb=apply_rotary_emb,
            activation=activation,
            drop_path=drop_path,
            norm_type=norm_type
        )

        self.pool_method = pool_method
        if pool_method == "average":
            self.pool = AverageReducer()
        elif pool_method == "bert":
            self.pool = BertPooler(dim)
        else:
            raise ValueError(f"Unknown pool method: {pool_method}")

        self.classifier = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(
                dim=dim,
                hidden_dim=mlp_dim,
                dropout=0.0,
                activation=activation,
                output_dim=num_classes
            )
        )

    def forward(self, inputs: Union[str, List[str], Tuple], return_embeddings: bool = False):
        if isinstance(inputs, str) or (isinstance(inputs, list) and isinstance(inputs[0], str)):
            if self.toxic_bert_encoder is None:
                raise ValueError("You must specify detoxify_model when passing strings as input.")

            with torch.no_grad():
                embeddings, attention_mask = self.toxic_bert_encoder(inputs, return_attention_mask_too=True)
        else:
            embeddings, attention_mask = inputs

        return self.forward_tensor(embeddings, attention_mask, return_embeddings)

    def forward_tensor(
            self,
            embeddings: torch.Tensor,
            attention_mask: torch.Tensor,
            return_embeddings: bool = False
    ):
        embeddings = self.transformer(embeddings, mask=attention_mask)

        if return_embeddings:
            return embeddings

        if self.pool_method == "average":
            embeddings = self.pool(embeddings, attention_mask)
        else:
            embeddings = self.pool(embeddings)

        logits = self.classifier(embeddings)
        return logits


if __name__ == "__main__":
    model = SexistBert(device="cuda", num_classes=1, pool_method="bert").cuda()

    data_loader = DataLoader(batch_size=16)
    for batch in tqdm(data_loader):
        text = list(batch['text'].values)
        y = model(text, return_embeddings=True)
