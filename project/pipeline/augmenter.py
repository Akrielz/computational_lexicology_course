from typing import Literal

import numpy as np

from project.eda.augmentations import load_teda, load_translator, \
    load_wordnet, load_context_aug, synonym_replacement_augmentation, \
    translation_augmentation, wordnet_augmentation, word_context_augmentation

AugmentationMethod = Literal["teda", "translation", "wordnet", "context", "random", "none"]


class TextAugmenter:
    def __init__(self):
        self.teda = load_teda()
        self.translator = load_translator()
        self.twordnet = load_wordnet()
        self.context_aug = load_context_aug()

        self.augmentation_methods = ["teda", "translation", "wordnet", "context"]

    def augment_text(
            self,
            text: str,
            method: AugmentationMethod = "random"
    ):
        if method == "random":
            method = np.random.choice(["teda", "translation", "wordnet", "context"])

        if method == "teda":
            return synonym_replacement_augmentation(text, self.teda)
        elif method == "translation":
            return translation_augmentation(text, self.translator)
        elif method == "wordnet":
            return wordnet_augmentation(text, self.twordnet)
        elif method == "context":
            return word_context_augmentation(text, self.context_aug)

        return text

    def __call__(
            self,
            text: str,
            method: AugmentationMethod = "random"
    ):
        return self.augment_text(text)