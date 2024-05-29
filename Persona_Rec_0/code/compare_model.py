import math
import os
import numpy
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import torch
import model


# ROOT_FOLDER = "../checkpoint/SIGIR2023"
ROOT_FOLDER = "checkpoint/SIGIR2023"
# DATASET_LIST = ["amazon_cds", "amazon_electronic", "amazon_beauty"]
DATASET_LIST = ["amazon_cds", "amazon_electronic"]

# TYPE_LIST = ["meta_ood_ocsvm", "meta_ood_lof", "meta_random", "meta_ood", "meta_ood_uncertainty", "meta_ood_uncertainty5", "meta_ood_uncertainty6"]
#
# TYPE_LIST_reverse = ["meta_random", "meta_ood", "meta_ood_ocsvm", "meta_ood_lof", "meta_ood_uncertainty", "meta_ood_uncertainty5", "meta_ood_uncertainty6"]

MODEL_LIST = ["din", "sasrec", "gru4rec"]

# log_filename = "best_auc.pkl"
# log_filename_ = "stage3_best_auc_ood.pkl"

for dataset in DATASET_LIST:
    print("{}{}{}".format("=" * 50, dataset, "=" * 50))
    for model in MODEL_LIST:
        print("{}{}{}".format("-" * 20, model, "-" * 20))

        # model_file = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), "base", "best_auc.pkl")
        model_file = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), "meta", "best_auc.pkl")
        model_ood_file = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), "meta_ood", "stage2_best_auc_ood.pkl")
        # model_ood_file = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), "meta_ood", "stage1_best_auc.pkl")
        # model_file = "/data/lvzheqi/MetaNetwork/MetaNetwork_RS_local_amazon_ood_seq10_from38_30u30i_gru_new_auc_vae_focal_from141_seed0_newuncertainty/checkpoint/SIGIR2023/amazon_cds_din/meta/best_auc.pkl"
        model_uncertainty_file = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), "meta_ood_uncertainty5", "stage3_best_auc_ood.pkl")
        # model_uncertainty_file = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), "meta_ood", "stage2_best_auc_ood.pkl")

        # model_dict = torch.load(model_file).state_dict()
        print(os.path.exists(model_file))
        model_dict = torch.load(model_file).state_dict()
        model_ood_dict = torch.load(model_ood_file).state_dict()
        model_uncertainty_dict = torch.load(model_uncertainty_file).state_dict()

        # count1 = 0
        # count2 = 0
        for key, value in model_dict.items():
            print("*" * 50)
            print(key)
            key_uncertainty = key
            # if key not in list(model_uncertainty_dict.keys()):
            #     key_uncertainty = "stage1" + key
            #     if key not in list(model_uncertainty_dict.keys()):
            #         print(key, "----")
            #         continue
            continue_ = False
            if key not in list(model_ood_dict.keys()):
                print("model_ood_dict not exists")
                continue_ = True
            if key not in list(model_uncertainty_dict.keys()):
                print("model_uncertainty_dict not exists")
                continue_ = True
            if continue_:
                continue
            # print(model_dict[key])
            # print(model_uncertainty_dict[key])
            # if model_dict[key] == model_uncertainty_dict[key]:
            print(torch.equal(model_dict[key], model_ood_dict[key_uncertainty]))
            print(torch.equal(model_dict[key], model_uncertainty_dict[key_uncertainty]))
            print(torch.equal(model_ood_dict[key_uncertainty], model_uncertainty_dict[key_uncertainty]))
            # if torch.equal(model_dict[key], model_uncertainty_dict[key_uncertainty]):
            #     print("=")
            # # elif model_dict[key] != model_uncertainty_dict[key]:
            # elif not torch.equal(model_dict[key], model_uncertainty_dict[key_uncertainty]):
            #     print("!=")
