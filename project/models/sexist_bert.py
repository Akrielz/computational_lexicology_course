from typing import Optional, Callable, Literal, List

from torch import nn
from vision_models_playground.components.attention import TransformerEncoder, FeedForward

from project.models.average_reducer import AverageReducer
from project.task_a.toxic_bert_encoder import ToxicBertEncoder


class SexistBert(nn.Module):
    def __init__(
            self,
            detoxify_model: str = "original",
            depth: int = 2,
            apply_rotary_emb: bool = True,
            activation: Optional[Callable] = None,
            drop_path: float = 0.0,
            norm_type: Literal['pre_norm', 'post_norm'] = "pre_norm",
            num_classes: int = 2,
            device: str = "cpu",
    ):
        super(SexistBert, self).__init__()
        self.toxic_bert_encoder = ToxicBertEncoder(detoxify_model, device)

        config = self.toxic_bert_encoder.bert_model.config
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

        self.average_reducer = AverageReducer()
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

    def forward(self, text: List[str], return_embeddings: bool = False):
        embeddings, attention_mask = self.toxic_bert_encoder(text, return_attention_mask_too=True)
        attention_mask = attention_mask.bool()
        embeddings = self.transformer(embeddings, mask=attention_mask)

        if return_embeddings:
            return embeddings

        embeddings = self.average_reducer(embeddings, attention_mask)
        logits = self.classifier(embeddings)
        return logits


if __name__ == "__main__":
    model = SexistBert(device="cuda").cuda()

    sentences = ["Let's go to the beach!", "Let's do it!", "The beach is a nice place to go."]
    logits = model(sentences)
    print(logits)
