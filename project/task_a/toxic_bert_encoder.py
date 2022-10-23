from typing import List

from torch import nn
from detoxify import Detoxify


class ToxicBertEncoder(nn.Module):
    def __init__(self, detoxify_model: str = "original", device: str = "cpu"):
        super(ToxicBertEncoder, self).__init__()

        detoxify = Detoxify(detoxify_model, device=device)

        self.tokenizer = detoxify.tokenizer
        self.bert_model = detoxify.model.bert

        # freeze bert model
        for param in self.bert_model.parameters():
            param.requires_grad = False

        self.device = device

    def forward(self, text: List[str], return_attention_mask_too: bool = False):
        tokenized = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        bert_output = self.bert_model(**tokenized)
        encoding = bert_output[0]

        if not return_attention_mask_too:
            return encoding

        return encoding, tokenized["attention_mask"]

    def output_dim(self):
        return self.bert_model.config.hidden_size


if __name__ == "__main__":
    toxic_bert_encoder = ToxicBertEncoder()
    encoding = toxic_bert_encoder(["hello world!"])
    print(encoding)