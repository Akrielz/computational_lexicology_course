from detoxify.detoxify import load_checkpoint, PRETRAINED_MODEL
from pprint import pprint


def get_toxic_bert_tokenizer(model_type="original", checkpoint=PRETRAINED_MODEL, device="cpu", huggingface_config_path=None):
    _, tokenizer, _ = load_checkpoint(
        model_type=model_type,
        checkpoint=checkpoint,
        device=device,
        huggingface_config_path=huggingface_config_path,
    )

    return tokenizer


if __name__ == "__main__":
    tokenizer = get_toxic_bert_tokenizer()
    sentences = ["Let's go to the beach!", "Let's do it!", "The beach is a nice place to go."]
    outputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    pprint(outputs)