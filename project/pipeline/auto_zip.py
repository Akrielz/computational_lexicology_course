import os

# get the files from the current directory

dir_path = "../data/semeval/submission_task_a/dev"
files = os.listdir(dir_path)

# remove extension from the files
files = [file.split(".")[0] for file in files]

# remove duplicates
files = list(set(files))

for file in files:
    file_path = os.path.join(dir_path, file)

    # check if the zip file exists
    if os.path.exists(f"{file_path}.zip"):
        continue

    # zip the file
    os.system(f"zip {file_path}.zip {file_path}.csv")