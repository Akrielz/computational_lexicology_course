import pandas as pd


def create_sexist_dataset_extended():
    # load the original dataset
    df_original = pd.read_csv("../data/train_all_tasks.csv")

    # load the extended dataset
    df_extended = pd.read_csv("../data/sexism_data.csv")

    # eliminate the columns "id" "dataset" "toxicity" "of_id" from df_extended
    df_extended = df_extended.drop(columns=["id", "dataset", "toxicity", "of_id"])

    # remap the "sexist" labels from df_extended to "not sexist" and "sexist"
    df_extended["sexist"] = df_extended["sexist"].map({False: "not sexist", True: "sexist"})

    # rename the column from "sexist" to "label_sexist"
    df_extended = df_extended.rename(columns={"sexist": "label_sexist"})

    # concatenate the two datasets
    df = pd.concat([df_original, df_extended], ignore_index=True)

    # save the dataframe as csv
    df.to_csv("../data/train_sexism.csv", index=False)


if __name__ == "__main__":
    create_sexist_dataset_extended()