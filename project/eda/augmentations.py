from googletrans import Translator
from textaugment import Wordnet, EDA
import nlpaug.augmenter.word as naw


def translate_text(texts, source_language, target_language, translator):
    translated_texts = translator.translate(texts, dest=target_language, src=source_language).text
    return translated_texts


def load_wordnet(v=False, n=True, p=0.5) -> Wordnet:
    return Wordnet(v=v, n=n, p=p)


def load_teda() -> EDA:
    return EDA()


def load_translator():
    return Translator()


def load_context_aug():
    return naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")


def synonym_replacement_augmentation(text: str, teda: EDA) -> str:
    """
    :return: randomly replace words with their synonyms in order to help it generalize
    """
    return teda.synonym_replacement(text)


def random_insertion_augmentation(text: str, teda: EDA) -> str:
    """
    :return: randomly insert a word in a text, thus inducing a bit of redundancy
    """
    return teda.random_insertion(text)


def random_deletion_augmentation(text: str, proba: float = 0.2, teda: EDA = None) -> str:
    """
    :return: randomly delete word a given probability proba
    """
    return teda.random_deletion(text, p=proba)


def random_swap_augmentation(text: str, teda: EDA) -> str:
    """
    beware this is dangerous if you are not using a contextual model because you create bias!
    :return: randomly swap words in a text
    """
    return teda.random_swap(text)


def word_context_augmentation(text: str, aug: naw.ContextualWordEmbsAug) -> str:
    """
    :return: text augmented using WordNet pipeline
    """
    augmented_text = aug.augment(text)
    return augmented_text[0]


def wordnet_augmentation(text: str, twordnet: Wordnet) -> str:
    """
    Should be used only for augmenting english texts
    :return: text augmented using WordNet pipeline
    """
    return twordnet.augment(text)


def translation_augmentation(text: str, translator, src="en") -> str:
    """
    :return: the original text after a mirrored translation
    """
    if src == "en":
        dst = "fr"
    else:
        dst = "en"

    translated = translate_text(text, src, dst, translator)
    re_translated = translate_text(translated, dst, src, translator)
    return re_translated


def main():
    text = "Hi everyone! Did you enjoy the show from last night? :D"
    print("Original text: \n", text)

    teda = load_teda()
    translator = Translator()
    twordnet = load_wordnet()
    aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute")

    print("\nsynonym_replacement_augmentation:\n", synonym_replacement_augmentation(text, teda))
    print("\nrandom_deletion_augmentation:\n", random_deletion_augmentation(text, 0.2, teda))
    print("\nrandom_swap_augmentation:\n", random_swap_augmentation(text, teda))
    print("\nrandom_insertion_augmentation:\n", random_insertion_augmentation(text, teda))
    print("\ntranslation_augmentation:\n", translation_augmentation(text, translator))
    print("\nwordnet_augmentation:\n", wordnet_augmentation(text, twordnet))
    print("\nword_context_augmentation:\n", word_context_augmentation(text, aug))


if __name__ == "__main__":
    main()
