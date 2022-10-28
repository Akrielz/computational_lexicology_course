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


def process_downloaded_twits(path):
    # read the saved csv
    df = pd.read_csv(path)

    # add column with sexist label
    df["label_sexist"] = "sexist"

    return df


def process_compliments_benevolent():
    return process_downloaded_twits("../data/compliments/benevolent_sexist_downloaded.csv")


def process_compliments_hostile():
    return process_downloaded_twits("../data/compliments/hostile_sexist_downloaded.csv")


def process_hate_speech():
    return process_downloaded_twits("../data/hate_speech/NAACL_SRW_2016_downloaded.csv")


def process_exist_test():
    # load df
    df = pd.read_csv("../data/exist/test/exist_2021_test_labeled.tsv", sep="\t")

    # keep only rows with language in Englsih
    df = df[df["language"] == "en"]

    # keep only text and task1
    df = df[["text", "task1"]]

    # rename the columns
    df = df.rename(columns={"task1": "label_sexist"})

    # remap the "non-sexist" to "not sexist
    df["label_sexist"] = df["label_sexist"].map({"non-sexist": "not sexist", "sexist": "sexist"})

    return df


def process_exist_train():
    # load df
    df = pd.read_csv("../data/exist/test/exist_2021_test_labeled.tsv", sep="\t")

    # keep only rows with language in Englsih
    df = df[df["language"] == "en"]

    # keep only text and task1
    df = df[["text", "task1"]]

    # rename the columns
    df = df.rename(columns={"task1": "label_sexist"})

    # remap the "non-sexist" to "not sexist
    df["label_sexist"] = df["label_sexist"].map({"non-sexist": "not sexist", "sexist": "sexist"})

    return df


def process_twitter_analysis_train():
    # load the original dataset
    df = pd.read_csv("../data/twitter_analysis/train.csv")

    # keep only columns "tweet" and "label"
    df = df[["tweet", "label"]]

    # rename the columns
    df = df.rename(columns={"tweet": "text", "label": "label_sexist"})

    # remap the 0 to "not sexist" and 1 to "sexist"
    df["label_sexist"] = df["label_sexist"].map({0: "not sexist", 1: "sexist"})

    return df


def process_workplace():
    # load dataset project/data/workplace/ise_sexist_data_labeling.xlsx
    df = pd.read_excel("../data/workplace/ise_sexist_data_labeling.xlsx")

    # rename the columns "Sentences" to "text" and "Label" to "label_sexist"
    df = df.rename(columns={"Sentences": "text", "Label": "label_sexist"})

    # remap the 0 to "not sexist" and 1 to "sexist"
    df["label_sexist"] = df["label_sexist"].map({0: "not sexist", 1: "sexist"})

    return df


def create_sexist_dataset_extended():
    # load the datasets
    df_list = []

    # df_list.append(process_original())
    df_list.append(process_csmb())
    # df_list.append(process_exist_test())
    # df_list.append(process_exist_train())
    # df_list.append(process_twitter_analysis_train())
    # df_list.append(process_workplace())
    # df_list.append(process_hate_speech())
    # df_list.append(process_compliments_benevolent())
    # df_list.append(process_compliments_hostile())

    # concatenate the dataframes
    df = pd.concat(df_list)

    # replace any @user with [USER]
    df["text"] = df["text"].str.replace("@\w+", "[USER]", regex=True)

    # replace all urls with [URL]
    df["text"] = df["text"].str.replace("http\S+", "[URL]", regex=True)
    df["text"] = df["text"].str.replace("https\S+", "[URL]", regex=True)
    df["text"] = df["text"].str.replace("www\S+", "[URL]", regex=True)

    # drop duplicates
    df = df.drop_duplicates()

    # drop all rows that are shorter than 2 but no longer than 400
    df = df[df["text"].str.len() > 2]
    df = df[df["text"].str.len() < 400]

    # save the dataframe to csv
    df.to_csv("../data/custom/train_sexist.csv", index=False)


if __name__ == "__main__":
    create_sexist_dataset_extended()
