import pandas as pd
from neuspell import BertChecker


def clean_dataset(text_column: "str", csv_file: "str", output_file: "str"):
    # read the data
    df = pd.read_csv(csv_file)

    # init spell checker
    checker = BertChecker()
    checker.from_pretrained()

    # clean the data
    corrected_text = checker.correct_strings(df[text_column].values.tolist())
    df[text_column] = corrected_text

    # save the data
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    clean_dataset(text_column="text", csv_file="../data/semeval/train_all_tasks.csv", output_file="../data/custom/train_all_tasks_corrected.csv")