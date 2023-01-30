import warnings

from project.task_a.train_sexist_bert_pretrained_dual import build_model as build_model_sexist
from project.task_b.sexist_bert_train import build_model as build_model_category
from project.task_c.sexist_bert_train import build_model as build_model_specific


def no_warnings_wrapper(func):
    def wrapper(*args, **kwargs):
        from transformers import logging
        logging.set_verbosity_error()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = func(*args, **kwargs)
            logging.set_verbosity_info()
            return res

    return wrapper


CHECKPOINT_PATHS = {
    "binary": "../trained_agents/sexist_bert_pretrained_dual_a_original.pt",
    "category": "../trained_agents/sexist_bert_pretrained_b_e_5.pt",
    "specific": "../trained_agents/sexist_bert_pretrained_c_e_0.pt",
}

BUILD_MODEL = {
    "binary": no_warnings_wrapper(build_model_sexist),
    "category": no_warnings_wrapper(build_model_category),
    "specific": no_warnings_wrapper(build_model_specific),
}

BINARY_TO_LABEL = {
    "not sexist": 0,
    "sexist": 1,
}

LABEL_TO_SEXIST = {v: k for k, v in BINARY_TO_LABEL.items()}

CATEGORY_TO_LABEL = {
    '1. threats, plans to harm and incitement': 0,
    '2. derogation': 1,
    '3. animosity': 2,
    '4. prejudiced discussions': 3
}

LABEL_TO_CATEGORY = {v: k for k, v in CATEGORY_TO_LABEL.items()}

SPECIFIC_TO_LABEL = {
    '1.1 threats of harm': 0,
    '1.2 incitement and encouragement of harm': 1,
    '2.1 descriptive attacks': 2,
    '2.2 aggressive and emotive attacks': 3,
    '2.3 dehumanising attacks & overt sexual objectification': 4,
    '3.1 casual use of gendered slurs, profanities, and insults': 5,
    '3.2 immutable gender differences and gender stereotypes': 6,
    '3.3 backhanded gendered compliments': 7,
    '3.4 condescending explanations or unwelcome advice': 8,
    '4.1 supporting mistreatment of individual women': 9,
    '4.2 supporting systemic discrimination against women as a group': 10,
}

LABEL_TO_SPECIFIC = {v: k for k, v in SPECIFIC_TO_LABEL.items()}

MAPPING = {
    "binary": BINARY_TO_LABEL,
    "category": CATEGORY_TO_LABEL,
    "specific": SPECIFIC_TO_LABEL
}

INVERSE_MAPPING = {
    "binary": LABEL_TO_SEXIST,
    "category": LABEL_TO_CATEGORY,
    "specific": LABEL_TO_SPECIFIC
}
