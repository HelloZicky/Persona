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
# MODEL_LIST = ["din", "gru4rec", "sasrec"]
MODEL_LIST = ["gru4rec", "sasrec"]

TYPE_LIST = ["base", "base_finetune", "meta",
             "meta_grad", "meta_grad_gru",
             "base_group_finetune_long_tail_2", "base_group_finetune_long_tail_3",
             "base_group_finetune_long_tail_5", "base_group_finetune_long_tail_10",
             "base_group_finetune_grad_2", "base_group_finetune_grad_3",
             "base_group_finetune_grad_5", "base_group_finetune_grad_10",
             "base_group_finetune_grad_gru_2", "base_group_finetune_grad_gru_3",
             "base_group_finetune_grad_gru_5", "base_group_finetune_grad_gru_10",
             # "base_group_finetune_group_2", "base_group_finetune_group_3",
             # "base_group_finetune_group_5", "base_group_finetune_group_10",
             "meta_grad_gru_group_2", "meta_grad_gru_group_3",
             "meta_grad_gru_group_5", "meta_grad_gru_group_10",
             # "meta_grad_gru_center_group_2_clip_0.1", "meta_grad_gru_center_group_3_clip_0.1",
             # "meta_grad_gru_center_group_5_clip_0.1", "meta_grad_gru_center_group_10_clip_0.1",
             # "meta_grad_gru_center_group_2_clip_0.5", "meta_grad_gru_center_group_3_clip_0.5",
             # "meta_grad_gru_center_group_5_clip_0.5", "meta_grad_gru_center_group_10_clip_0.5",
             # "meta_grad_gru_center_group_2_clip_1.0", "meta_grad_gru_center_group_3_clip_1.0",
             # "meta_grad_gru_center_group_5_clip_1.0", "meta_grad_gru_center_group_10_clip_1.0",
             # "meta_grad_gru_center_group_2_clip_5.0", "meta_grad_gru_center_group_3_clip_5.0",
             # "meta_grad_gru_center_group_5_clip_5.0", "meta_grad_gru_center_group_10_clip_5.0",
             "meta_long_tail_center_2_clip_1.0", "meta_long_tail_center_3_clip_1.0",
             "meta_long_tail_center_5_clip_1.0", "meta_long_tail_center_10_clip_1.0",
             "meta_grad_center_2_clip_1.0", "meta_grad_center_3_clip_1.0",
             "meta_grad_center_5_clip_1.0", "meta_grad_center_10_clip_1.0",
             "meta_grad_gru_center_2_clip_1.0", "meta_grad_gru_center_3_clip_1.0",
             "meta_grad_gru_center_5_clip_1.0", "meta_grad_gru_center_10_clip_1.0",
             # "meta_grad_gru_center_group_2_clip_1.0", "meta_grad_gru_center_group_3_clip_1.0",
             # "meta_grad_gru_center_group_5_clip_1.0", "meta_grad_gru_center_group_10_clip_1.0",
             ]

NAME_LIST = ["Base", "Finetune", "DUET",
             "DUET (Grad.)", "DUET (Grad.GRU)",
             "Longtail-Finetune-2", "Longtail-Finetune-3",
             "Longtail-Finetune-5", "Longtail-Finetune-10",
             "Grad-Finetune-2", "Grad-Finetune-3",
             "Grad-Finetune-5", "Grad-Finetune-10",
             "GradGRU-Finetune-2", "GradGRU-Finetune-3",
             "GradGRU-Finetune-5", "GradGRU-Finetune-10",
             # "Finetune-2", "Finetune-3",
             # "Finetune-5", "Finetune-10",
             "DUET (Grad.GRU)-base-2-1.0", "DUET (Grad.GRU)-base-3-1.0",
             "DUET (Grad.GRU)-base-5-1.0", "DUET (Grad.GRU)-base-10-1.0",
             # "DUET (Grad.GRU)-group-2-0.1", "DUET (Grad.GRU)-group-3-0.1",
             # "DUET (Grad.GRU)-group-5-0.1", "DUET (Grad.GRU)-group-10-0.1",
             # "DUET (Grad.GRU)-group-2-0.5", "DUET (Grad.GRU)-group-3-0.5",
             # "DUET (Grad.GRU)-group-5-0.5", "DUET (Grad.GRU)-group-10-0.5",
             # "DUET (Grad.GRU)-group-2-1.0", "DUET (Grad.GRU)-group-3-1.0",
             # "DUET (Grad.GRU)-group-5-1.0", "DUET (Grad.GRU)-group-10-1.0",
             # "DUET (Grad.GRU)-group-2-5.0", "DUET (Grad.GRU)-group-3-5.0",
             # "DUET (Grad.GRU)-group-5-5.0", "DUET (Grad.GRU)-group-10-5.0",
             "DUET (Grad.GRU)-long_tail-2-1.0", "DUET (Grad.GRU)-long_tail-3-1.0",
             "DUET (Grad.GRU)-long_tail-5-1.0", "DUET (Grad.GRU)-long_tail-10-1.0",
             "DUET (Grad.GRU)-grad-2-1.0", "DUET (Grad.GRU)-grad-3-1.0",
             "DUET (Grad.GRU)-grad-5-1.0", "DUET (Grad.GRU)-grad-10-1.0",
             "DUET (Grad.GRU)-grad_gru-2-1.0", "DUET (Grad.GRU)-grad_gru-3-1.0",
             "DUET (Grad.GRU)-grad_gru-5-1.0", "DUET (Grad.GRU)-grad_gru-10-1.0",
             ]

result_dict = {}
result_txt = os.path.join(ROOT_FOLDER, "result_ood_overall.txt")
result_csv = os.path.join(ROOT_FOLDER, "result_ood1_overall.csv")

with open(result_txt, "w+") as txt_writer, open(result_csv, "w+") as csv_writer:
    for dataset in DATASET_LIST:
        print("=" * 50, file=txt_writer)
        print(dataset, file=txt_writer)
        print(dataset, file=csv_writer)
        # print("\n", file=csv_writer)
        print(file=csv_writer)
        print("-" * 50, file=txt_writer)
        for model in MODEL_LIST:
            for (type, name) in zip(TYPE_LIST, NAME_LIST):
                # root_folder = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type)
                # log_file = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type, log_filename)
                if type in ["base", "meta",
                            "meta_grad", "meta_grad_gru",
                            "base_group_finetune_long_tail_2", "base_group_finetune_long_tail_3",
                            "base_group_finetune_long_tail_5", "base_group_finetune_long_tail_10",
                            "base_group_finetune_grad_2", "base_group_finetune_grad_3",
                            "base_group_finetune_grad_5", "base_group_finetune_grad_10",
                            "base_group_finetune_grad_gru_2", "base_group_finetune_grad_gru_3",
                            "base_group_finetune_grad_gru_5", "base_group_finetune_grad_gru_10",
                            "base_group_finetune_group_2", "base_group_finetune_group_3",
                            "base_group_finetune_group_5", "base_group_finetune_group_10",
                            "meta_grad_gru_group_2", "meta_grad_gru_group_3",
                            "meta_grad_gru_group_5", "meta_grad_gru_group_10",
                            # "meta_grad_gru_center_group_2_clip_0.1", "meta_grad_gru_center_group_3_clip_0.1",
                            # "meta_grad_gru_center_group_5_clip_0.1", "meta_grad_gru_center_group_10_clip_0.1",
                            # "meta_grad_gru_center_group_2_clip_0.5", "meta_grad_gru_center_group_3_clip_0.5",
                            # "meta_grad_gru_center_group_5_clip_0.5", "meta_grad_gru_center_group_10_clip_0.5",
                            # "meta_grad_gru_center_group_2_clip_1.0", "meta_grad_gru_center_group_3_clip_1.0",
                            # "meta_grad_gru_center_group_5_clip_1.0", "meta_grad_gru_center_group_10_clip_1.0",
                            # "meta_grad_gru_center_group_2_clip_5.0", "meta_grad_gru_center_group_3_clip_5.0",
                            # "meta_grad_gru_center_group_5_clip_5.0", "meta_grad_gru_center_group_10_clip_5.0",
                            "meta_long_tail_center_2_clip_1.0", "meta_long_tail_center_3_clip_1.0",
                            "meta_long_tail_center_5_clip_1.0", "meta_long_tail_center_10_clip_1.0",
                            "meta_grad_center_2_clip_1.0", "meta_grad_center_3_clip_1.0",
                            "meta_grad_center_5_clip_1.0", "meta_grad_center_10_clip_1.0",
                            "meta_grad_gru_center_2_clip_1.0", "meta_grad_gru_center_3_clip_1.0",
                            "meta_grad_gru_center_5_clip_1.0", "meta_grad_gru_center_10_clip_1.0",
                            ]:
                    max_auc = 0
                    # log_file = [os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type, "log_ood.txt")]
                    log_file = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type, "log_ood.txt")
                elif type == "base_finetune":
                    log_file = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type, "log_overall.txt")

                auc_list, auc_user_list, ndcg5_list, hr5_list, ndcg10_list, hr10_list, ndcg20_list, hr20_list = \
                    [], [], [], [], [], [], [], []

                if not os.path.exists(log_file):
                    # print(log_file)
                    print("{:8s}\t{:30s}".format(
                        model,
                        name,
                    ),
                        sep="\t",
                        # file=[txt_writer, csv_writer],
                        file=txt_writer,
                    )
                    print("{:8s}\t{:30s}".format(
                        model,
                        name,
                    ),
                        sep="\t",
                        # file=[txt_writer, csv_writer],
                        file=csv_writer,
                    )
                    continue

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
                    file=csv_writer,
                )
        print(file=csv_writer)
    # for dataset in DATASET_LIST:
    #     print("=" * 50, file=txt_writer)
    #     print(dataset, file=txt_writer)
    #     print("-" * 50, file=txt_writer)
    #     for model in MODEL_LIST:
    #         for (type, name) in zip(TYPE_LIST, NAME_LIST):