from detoxify.detoxify import load_checkpoint
from pprint import pprint

from transformers import AutoTokenizer


def get_toxic_bert_tokenizer(model_type="original", checkpoint=None, huggingface_config_path=None):
    _, tokenizer, _ = load_checkpoint(
        model_type=model_type,
        checkpoint=checkpoint,
        device="cpu",
        huggingface_config_path=huggingface_config_path,
    )

    return tokenizer


def get_bert_tokenizer(model_type="bert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(model_type, use_fast=False)
    return tokenizer


if __name__ == "__main__":
    tokenizer = get_toxic_bert_tokenizer()
    sentences = ["Let's go to the beach!", "Let's do it!", "The beach is a nice place to go."]
    outputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    pprint(outputs)

    tokenizer = get_bert_tokenizer()
    outputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    pprint(outputs)