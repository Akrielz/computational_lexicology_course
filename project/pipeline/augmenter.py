from typing import Literal, List, Union

import numpy as np
from tqdm import tqdm

from project.pipeline.augmentations import load_teda, load_back_translator, \
    load_wordnet, load_context_aug, synonym_replacement_augmentation, \
    translation_augmentation, wordnet_augmentation, word_context_augmentation

AugmentationMethod = Literal["teda", "translation", "wordnet", "context", "random", "none"]


class TextAugmenter:
    def __init__(self, device="cpu"):
        self.teda = load_teda()
        self.translator = load_back_translator(device)
        self.twordnet = load_wordnet(device)
        self.context_aug = load_context_aug(device)

        self.slow_augmentation_methods = ["translation"]
        self.medium_augmentation_methods = ["context"]
        self.fast_augmentation_methods = ["teda", "wordnet"]

        self.all_augmentation_methods = self.slow_augmentation_methods + self.medium_augmentation_methods + \
                                    self.fast_augmentation_methods

    def get_augmentation_methods(self, speed: Literal["slow", "medium", "fast"]):
        methods = []
        if speed == "slow":
            methods.extend(self.slow_augmentation_methods)
            return methods

        elif speed == "medium":
            methods.extend(self.medium_augmentation_methods)
            return methods

        elif speed == "fast":
            methods.extend(self.fast_augmentation_methods)
            return methods

    def augment_text(
            self,
            text: str,
            method: AugmentationMethod = "random",
            speed: Literal["fast", "medium", "slow"] = "fast"
    ):
        if method == "random":
            method = np.random.choice(self.get_augmentation_methods(speed))

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
            text: Union[str, List[str]],
            method: AugmentationMethod = "random",
            progress_bar: bool = False
    ):
        if isinstance(text, str):
            return self.augment_text(text, method=method)

        progress_bar = tqdm(text) if progress_bar else text
        return [self.augment_text(t, method=method) for t in progress_bar]