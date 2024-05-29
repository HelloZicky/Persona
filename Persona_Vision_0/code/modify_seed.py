import re
import os
from tqdm import tqdm

# scripts_folder = "scripts_backup"
scripts_folder = "main_new"
old_str = "utils.setup_seed(2023)"
new_str = "utils.setup_seed(0)"


for root, folders, files in os.walk(scripts_folder):
    # for folder in folders:
    #     folder_path = os.path.join(root, folder)
    #     print(folder_path)
    # for file in tqdm(files):
    for file in files:
        suffix = file.split(".")[-1]

        file_path = os.path.join(root, file)
        # if suffix != "sh" or len(file_path.split("/")) != 4:
        #     continue
        if suffix != "py":
            continue
        # print(file_path)
        folder = file_path.split("/")[1]
        # with open(file_path, "r+") as reader, open(file_path, "w+") as writer:
        records = []
        with open(file_path, "r+") as reader:
            for line in reader:
                # print(line)
                if re.match(old_str, line):
                    print("---Modified--------", file_path)
                new_line = line.strip().replace(old_str, new_str)
                records.append(new_line)
                # if file_path.split("/")[1] == "new"
                #     new_line = new_line.\
                #         replace("export CUDA_VISIBLE_DEVICES=0", "export CUDA_VISIBLE_DEVICES=2").\
                #         replace("export CUDA_VISIBLE_DEVICES=7", "export CUDA_VISIBLE_DEVICES=2")
                # print(new_line, file=reader)

        with open(file_path, "w+") as writer:
            for line in records:
                print(line, file=writer)
