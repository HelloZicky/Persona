import os

dataset = "/mnt/lzq/MetaNetwork/data/ml-1m/nips2022_dataset"
train_file = os.path.join(dataset, "train.txt")
test_file = os.path.join(dataset, "test.txt")

for file in [train_file, test_file]:
    count = 0
    with open(file, "r") as reader:
        for line in reader:
            count += 1
    print(file)
    print(count)
