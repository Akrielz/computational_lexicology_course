from typing import Literal, List, Union

import numpy as np
import torch

from project.pipeline.classic_tokenizers import get_bert_tokenizer
from project.wrapper.info import CHECKPOINT_PATHS, BUILD_MODEL, INVERSE_MAPPING


class SexistBert:
    def __init__(
            self,
            model_type: Literal["binary", "category", "specific"],
            device: torch.device = "cpu",
            batch_size: int = 32,
    ):
        # Save data
        self.model_type = model_type
        self.device = device
        self.batch_size = batch_size

        # Load tokenizer
        self.tokenizer = get_bert_tokenizer()

        # Load model
        self._load_models()

    def _load_models(self):
        binary_model_path = CHECKPOINT_PATHS["binary"]
        self.binary_model = BUILD_MODEL['binary'](binary_model_path)
        self.binary_model.to(self.device)
        self.binary_model.eval()

        if self.model_type == "binary":
            return

        category_model_path = CHECKPOINT_PATHS[self.model_type]
        self.sexist_model = BUILD_MODEL[self.model_type](category_model_path)
        self.sexist_model.to(self.device)
        self.sexist_model.eval()

    def _iter_batch_indices(self, text: List[str]):
        for i in range(0, len(text), self.batch_size):
            yield text[i:i + self.batch_size]

    @torch.no_grad()
    def _apply_model(
            self,
            model: torch.nn.Module,
            text: List[str]
    ):
        # Iterate the tokenized text in batches
        predictions = []
        for batch in self._iter_batch_indices(text):
            # Tokenize the batch
            batch_tokenized = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            batch_tokenized.to(self.device)

            # Apply the model
            batch_prediction = model(**batch_tokenized)[0]
            batch_prediction = torch.argmax(batch_prediction, dim=1).detach().cpu().numpy()
            predictions.extend(batch_prediction)

        # Compute the labels
        predictions = np.array(predictions)

        # Apply the inverse label mapping
        model_type = "binary" if model is self.binary_model else self.model_type
        predictions = [INVERSE_MAPPING[model_type][label] for label in predictions]
        return predictions

    def predict(self, text: Union[str, List[str]]):
        # Prepare the text
        if type(text) == str:
            text = [text]

        assert len(text) > 0, "The text must be a non-empty string or a list of non-empty strings."

        # Apply the binary model
        binary_predictions = self._apply_model(self.binary_model, text)

        if self.model_type == "binary":
            return binary_predictions

        # Get only the sexist ones
        sexist_text = [text[i] for i in range(len(text)) if binary_predictions[i] == "sexist"]

        # Make sure we apply the category only if we have sexist samples
        if len(sexist_text) == 0:
            return binary_predictions

        # Apply the sexist model to determine the type of sexism
        sexist_predictions = self._apply_model(self.sexist_model, sexist_text)

        # Update the predictions
        final_predictions = []
        for i in range(len(binary_predictions)):
            if binary_predictions[i] == "sexist":
                final_predictions.append(sexist_predictions.pop(0))
            else:
                final_predictions.append(binary_predictions[i])

        return final_predictions


if __name__ == "__main__":
    sexist_bert = SexistBert("specific", batch_size=2)
    print(sexist_bert.predict(["hello world!", "women are the worst", "all women should die!", "hello world 2!"]))