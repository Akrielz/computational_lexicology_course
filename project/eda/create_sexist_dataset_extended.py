import pandas as pd


def process_original():
    # load the original dataset
    df = pd.read_csv("../data/semeval/train_all_tasks.csv")

    # keep only columns "text" and "label_sexist"
    df = df[["text", "label_sexist"]]

    return df


def process_csmb():
    # load the extended dataset
    df = pd.read_csv("../data/cmsb/sexism_data.csv")

    # eliminate the columns "id" "dataset" "toxicity" "of_id" from df
    df = df.drop(columns=["id", "dataset", "toxicity", "of_id"])

    # remap the "sexist" labels from df to "not sexist" and "sexist"
    df["sexist"] = df["sexist"].map({False: "not sexist", True: "sexist"})

    # rename the column from "sexist" to "label_sexist"
    df = df.rename(columns={"sexist": "label_sexist"})

    return df


def process_compliments():
    # read .../data/compliments/benevolent_sexist.tsv
    pass


def process_conv_abuse():
    # load the dataset
    df = pd.read_csv("../data/conv_abuse/conv_abuse_emnlp_full.csv")

    # keep only the columns "text" and "label"
    df = df[["text", "label"]]

    # rename the column



def create_sexist_dataset_extended():
    # load the original dataset
    df_original = process_original()
    df_csmb = process_csmb()


    df = pd.concat([df_original, df_extended], ignore_index=True)

    # save the dataframe as csv
    df.to_csv("../data/custom/train_sexism.csv", index=False)


if __name__ == "__main__":
    create_sexist_dataset_extended()