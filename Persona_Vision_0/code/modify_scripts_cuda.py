import random
import re
import os
from tqdm import tqdm

# scripts_folder = "scripts_backup"
# scripts_folder = "scripts_1205"
scripts_folder = "scripts"
# std_str = "CUDA_VISIBLE_DEVICES="
std_str = "export CUDA_VISIBLE_DEVICES="
# old_str = "CUDA_VISIBLE_DEVICES=3"
# new_str = "CUDA_VISIBLE_DEVICES="


for root, folders, files in os.walk(scripts_folder):
    # for folder in folders:
    #     folder_path = os.path.join(root, folder)
    #     print(folder_path)
    # for file in tqdm(files):
    for file in files:
        suffix = file.split(".")[-1]

        file_path = os.path.join(root, file)
        if suffix != "sh" or len(file_path.split("/")) != 4:
            continue
        # print(file_path)
        folder = file_path.split("/")[1]
        # with open(file_path, "r+") as reader, open(file_path, "w+") as writer:
        records = []
        old_str = ""
        new_str = ""
        with open(file_path, "r+") as reader:
            for line in reader:
                # print(line)
                # if re.match(old_str, line):
                # print(line)
                if re.match(std_str, line):
                    cuda_num = line.strip().split("=")[-1]
                    old_str = std_str + cuda_num
                    if cuda_num == 0 or cuda_num == 2:
                        if random.random() < 0.5:
                            cuda_num = 1
                        else:
                            cuda_num = 3
                        # cuda_num = 0
                    new_str = std_str + str(int(cuda_num) % 4)
                    print("---Modified--------", file_path)
                    # print(old_str)
                    # print(new_str)
                    # print(line.strip())
                    # print(line.strip().replace(old_str, new_str))
                    # print("-" * 50)
                # print("old_str ", old_str)
                # print("new_str ", new_str)
                new_line = line.strip().replace(old_str, new_str)
                    # replace(" --max_epoch=20", "").\
                    # replace(" --max_epoch=${MAX_EPOCH}", "").\
                    # replace("BATCH_SIZE=512", "BATCH_SIZE=1024").\
                    # replace('ARCH_CONF_FILE="amazon', 'ARCH_CONF_FILE="../amazon').\
                    # replace('ARCH_CONF_FILE="movielens', 'ARCH_CONF_FILE="../movielens')
                records.append(new_line)
                # if file_path.split("/")[1] == "new"
                #     new_line = new_line.\
                #         replace("export CUDA_VISIBLE_DEVICES=0", "export CUDA_VISIBLE_DEVICES=2").\
                #         replace("export CUDA_VISIBLE_DEVICES=7", "export CUDA_VISIBLE_DEVICES=2")
                # print(new_line, file=reader)

        with open(file_path, "w+") as writer:
            for line in records:
                print(line, file=writer)
