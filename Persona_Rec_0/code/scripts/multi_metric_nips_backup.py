import math
import os
import numpy
import numpy as np
from collections import defaultdict
import csv

ROOT_FOLDER = "../checkpoint/NIPS2023"
DATASET_LIST = ["amazon_beauty", "amazon_cds", "amazon_electronic",
                "douban_book", "douban_music",
                "movielens_100k", "movielens_1m"]
# MODEL_LIST = ["din", "sasrec", "gru4rec"]
MODEL_LIST = ["din", "gru4rec", "sasrec"]

TYPE_LIST = ["base", "base_finetune", "meta",
             "meta_grad", "meta_grad_gru",
             "meta_grad_gru_group_2", "meta_grad_gru_group_3",
             "meta_grad_gru_group_5", "meta_grad_gru_group_10",
             "meta_grad_gru_center_group_2_clip_0.5", "meta_grad_gru_center_group_3_clip_0.5",
             "meta_grad_gru_center_group_5_clip_0.5", "meta_grad_gru_center_group_10_clip_0.5",
             "meta_grad_gru_center_group_2_clip_1.0", "meta_grad_gru_center_group_3_clip_1.0",
             "meta_grad_gru_center_group_5_clip_1.0", "meta_grad_gru_center_group_10_clip_1.0",
             ]

NAME_LIST = ["Base", "Finetune", "DUET",
             "DUET (Grad.)", "DUET (Grad.GRU)",
             "DUET (Grad.GRU)-base-2-0.5", "DUET (Grad.GRU)-base-3-0.5",
             "DUET (Grad.GRU)-base-5-0.5", "DUET (Grad.GRU)-base-10-0.5",
             "DUET (Grad.GRU)-group-2-0.5", "DUET (Grad.GRU)-group-3-0.5",
             "DUET (Grad.GRU)-group-5-0.5", "DUET (Grad.GRU)-group-10-0.5",
             "DUET (Grad.GRU)-group-2-1.0", "DUET (Grad.GRU)-group-3-1.0",
             "DUET (Grad.GRU)-group-5-1.0", "DUET (Grad.GRU)-group-10-1.0",
             ]

result_dict = {}
result_txt = os.path.join(ROOT_FOLDER, "result_ood_overall.txt")
result_csv = os.path.join(ROOT_FOLDER, "result_ood1_overall.csv")

with open(result_txt, "w+") as txt_writer, open(result_csv, "w+") as csv_writer:
    for dataset in DATASET_LIST:
        print("=" * 50, file=txt_writer)
        print(dataset, file=txt_writer)
        print("-" * 50, file=txt_writer)
        for model in MODEL_LIST:
            for (type, name) in zip(TYPE_LIST, NAME_LIST):
                # root_folder = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type)
                # log_file = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type, log_filename)
                if type in ["base", "meta",
                            "meta_grad", "meta_grad_gru",
                            "meta_grad_gru_group_2", "meta_grad_gru_group_3",
                            "meta_grad_gru_group_5", "meta_grad_gru_group_10",
                            "meta_grad_gru_center_group_2_clip_0.5", "meta_grad_gru_center_group_3_clip_0.5",
                            "meta_grad_gru_center_group_5_clip_0.5", "meta_grad_gru_center_group_10_clip_0.5",
                            "meta_grad_gru_center_group_2_clip_1.0", "meta_grad_gru_center_group_3_clip_1.0",
                            "meta_grad_gru_center_group_5_clip_1.0", "meta_grad_gru_center_group_10_clip_1.0",
                            ]:
                    max_auc = 0
                    # log_file = [os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type, "log_ood.txt")]
                    log_file = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type, "log_ood.txt")
                elif type == "base_finetune":
                    log_file = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type, "log_overall.txt")

                if not os.path.exists(log_file):
                    # print(log_file)
                    continue

                auc_list, auc_user_list, ndcg5_list, hr5_list, ndcg10_list, hr10_list, ndcg20_list, hr20_list = \
                    [], [], [], [], [], [], [], []
                with open(log_file, "r+") as reader:
                    for index, line in enumerate(reader, 1):
                        auc = float(line.strip("\n").split(",")[2].split("=")[-1])
                        auc_user = float(line.strip("\n").split(",")[3].split("=")[-1])
                        # logloss = float(line.strip("\n").split(",")[4].split("=")[-1])
                        ndcg5 = float(line.strip("\n").split(",")[5].split("=")[-1])
                        hr5 = float(line.strip("\n").split(",")[6].split("=")[-1])
                        ndcg10 = float(line.strip("\n").split(",")[7].split("=")[-1])
                        hr10 = float(line.strip("\n").split(",")[8].split("=")[-1])
                        ndcg20 = float(line.strip("\n").split(",")[9].split("=")[-1])
                        hr20 = float(line.strip("\n").split(",")[10].split("=")[-1])

                        auc_list.append(auc)
                        auc_user_list.append(auc_user)
                        ndcg5_list.append(ndcg5)
                        hr5_list.append(hr5)
                        ndcg10_list.append(ndcg10)
                        hr10_list.append(hr10)
                        ndcg20_list.append(ndcg20)
                        hr20_list.append(hr20)

                # result_dict["{}_{}_{}".format(dataset, model, type)] = [
                #     max(auc_list),
                #     max(auc_user_list),
                #     max(ndcg5_list),
                #     max(hr5_list),
                #     max(ndcg10_list),
                #     max(hr10_list),
                #     max(ndcg20_list),
                #     max(hr20_list),
                # ]
                if type != "base_finetune":
                    # assert len(auc_user_list) == 20
                    if len(auc_user_list) % 20 != 0:
                        print("------", dataset, model, type, len(auc_user_list), "------", sep="\t")
                # print("\t".join(result_dict["{}_{}_{}".format(dataset, model, type)]))
                # print("{:6s}\t{:25s}\t{:8s}\t{:8s}\t{:8s}\t{:8s}\t{:8s}\t{:8s}\t{:8s}\t{:8s}".format(
                print("{:8s}\t{:30s}\t{:8s}\t{:8s}\t{:8s}\t{:8s}\t{:8s}\t{:8s}\t{:8s}".format(
                    model,
                    name,
                    # str(round(max(auc_list), 5)),
                    str(round(max(auc_user_list), 5)),
                    str(round(max(ndcg5_list), 5)),
                    str(round(max(hr5_list), 5)),
                    str(round(max(ndcg10_list), 5)),
                    str(round(max(hr10_list), 5)),
                    str(round(max(ndcg20_list), 5)),
                    str(round(max(hr20_list), 5)),
                    ),
                    sep="\t",
                    # file=[txt_writer, csv_writer],
                    file=txt_writer,
                )
    # for dataset in DATASET_LIST:
    #     print("=" * 50, file=txt_writer)
    #     print(dataset, file=txt_writer)
    #     print("-" * 50, file=txt_writer)
    #     for model in MODEL_LIST:
    #         for (type, name) in zip(TYPE_LIST, NAME_LIST):