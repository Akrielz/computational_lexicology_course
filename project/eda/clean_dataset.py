from tqdm import tqdm
from transformers import pipeline

from project.pipeline.data_loader import DataLoader


def clean_dataset(csv_file: str, output_file: str, text_column: str = "text"):
    # read the data
    data_loader = DataLoader(
        data_path=csv_file, batch_size=16, shuffle=False, balance_data_method="none", task_column_name=None
    )

    # init spell checker
    fix_spelling = pipeline("text2text-generation", model="oliverguhr/spelling-correction-english-base", device="cuda:0")

    corrected_texts = []
    for batch in tqdm(data_loader):
        text = list(batch["text"].values)
        corrected_text = fix_spelling(text, max_length=400)
        corrected_text = [item["generated_text"] for item in corrected_text]
        corrected_texts.extend(corrected_text)

    # copy the data_loader df
    df = data_loader.df.copy()

    # add the corrected texts
    df[text_column] = corrected_texts

    # save the data
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    clean_dataset(
        csv_file="../data/semeval/dev_task_a_entries.csv",
        output_file="../data/custom/dev_task_a_entries_corrected.csv",
        text_column="text",
    )
