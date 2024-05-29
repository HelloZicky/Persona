import math
import os
import numpy
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# ROOT_FOLDER = "../checkpoint/NIPS2022"
# ROOT_FOLDER = "../checkpoint/WWW2023"
ROOT_FOLDER = "../checkpoint/SIGIR2023"
ROOT_FOLDER_ = "../checkpoint/SIGIR2023_backup"
# ROOT_FOLDER = "../checkpoint/SIGIR2023_backup"
# DATASET_LIST = ["amazon_cds", "amazon_tv", "amazon_electronic", "amazon_beauty", "amazon_clothing", "amazon_books", "amazon_sports", "movielens", "movielens_100k", ]
DATASET_LIST = ["amazon_cds", "amazon_electronic", "amazon_beauty", "movielens"]
# TYPE_LIST = ["base", "base_finetune", "meta", "meta_gru", "meta_random", "meta_ood_ocsvm", "meta_ood_lof", "meta_ood_if", "meta_ood", "meta_ood_gru", "meta_ood_uncertainty"]
# TYPE_LIST = ["meta_random", "meta_ood", "meta_ood_uncertainty"]
# TYPE_LIST = ["meta_random", "meta_ood"]
TYPE_LIST = ["meta_random", "meta_ood", "meta_ood_ocsvm", "meta_ood_lof"]
# DATASET_LIST = ["movielens", "movielens_100k"]
# DATASET_LIST = ["amazon_beauty"]
# MODEL_LIST = ["din", "gru4rec", "sasrec"]
MODEL_LIST = ["din", "sasrec", "gru4rec"]

log_filename = "test.txt"
log_filename_ = "log_ood.txt"

result_file = os.path.join(ROOT_FOLDER, "request_curve.txt")
fig_file = os.path.join(ROOT_FOLDER, "request_curve.png")

# with open(result_file, "w+") as writer:
fig_folder = os.path.join(ROOT_FOLDER, "_figures")
plt.figure()
for dataset in DATASET_LIST:
    print("{}{}{}".format("=" * 50, dataset, "=" * 50))
    for model in MODEL_LIST:
        print("{}{}{}".format("-" * 20, model, "-" * 20))
        auc_dict = defaultdict(list)
        auc_user_dict = defaultdict(list)
        logloss_dict = defaultdict(list)
        ndcg5_dict = defaultdict(list)
        ndcg10_dict = defaultdict(list)
        ndcg20_dict = defaultdict(list)
        hr5_dict = defaultdict(list)
        hr10_dict = defaultdict(list)
        hr20_dict = defaultdict(list)
        rate_dict = defaultdict(list)

        # for type in TYPE_LIST:
        # for type, legend_name in zip(TYPE_LIST, ["DUET (Random)", "DUET (MR)", "DUET (MRU)"]):
        for type in TYPE_LIST:
            if type in ["meta_random", "meta_ood"]:
                log_file = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type, log_filename)
            elif type in ["meta_ood_ocsvm", "meta_ood_lof"]:
                log_file = os.path.join(ROOT_FOLDER_, "{}_{}".format(dataset, model), type, log_filename_)

            auc_list = []
            auc_user_list = []
            logloss_list = []
            ndcg5_list = []
            ndcg10_list = []
            ndcg20_list = []
            hr5_list = []
            hr10_list = []
            hr20_list = []
            request_num_list = []
            total_num_list = []
            rate_list = []
            if not os.path.exists(log_file):
                print(log_file)
            with open(log_file, "r+") as reader:
                for index, line in enumerate(reader, 1):
                    epoch_num = int(line.strip("\n").split(",")[0].split("=")[-1])
                    auc = float(line.strip("\n").split(",")[2].split("=")[-1])
                    auc_user = float(line.strip("\n").split(",")[3].split("=")[-1])
                    logloss = float(line.strip("\n").split(",")[4].split("=")[-1])
                    ndcg5 = float(line.strip("\n").split(",")[5].split("=")[-1])
                    hr5 = float(line.strip("\n").split(",")[6].split("=")[-1])
                    ndcg10 = float(line.strip("\n").split(",")[7].split("=")[-1])
                    hr10 = float(line.strip("\n").split(",")[8].split("=")[-1])
                    ndcg20 = float(line.strip("\n").split(",")[9].split("=")[-1])
                    hr20 = float(line.strip("\n").split(",")[10].split("=")[-1])
                    rate = float(line.strip("\n").split(",")[-1].split("=")[-1])

                    auc_list.append(auc)
                    auc_user_list.append(auc_user)
                    logloss_list.append(logloss)
                    ndcg5_list.append(ndcg5)
                    ndcg10_list.append(ndcg10)
                    ndcg20_list.append(ndcg20)
                    hr5_list.append(hr5)
                    hr10_list.append(hr10)
                    hr20_list.append(hr20)
                    rate_list.append(rate)
                    if type in ["meta_ood_ocsvm", "meta_ood_lof"] and index % 10 == 0:
                        auc_dict[type].append(max(auc_list))
                        auc_user_dict[type].append(max(auc_user_list))
                        logloss_dict[type].append(max(logloss_list))
                        ndcg5_dict[type].append(max(ndcg5_list))
                        ndcg10_dict[type].append(max(ndcg10_list))
                        ndcg20_dict[type].append(max(ndcg20_list))
                        hr5_dict[type].append(max(hr5_list))
                        hr10_dict[type].append(max(hr10_list))
                        hr20_dict[type].append(max(hr20_list))
                        rate_dict[type].append(rate_list[auc_user_dict.index(max(auc_user_dict))])

                        auc_list = []
                        auc_user_list = []
                        logloss_list = []
                        ndcg5_list = []
                        ndcg10_list = []
                        ndcg20_list = []
                        hr5_list = []
                        hr10_list = []
                        hr20_list = []
                        rate_list = []

                    if rate == float(100):
                        break
            if type in ["meta_random", "meta_ood"]:
                auc_dict[type] = auc_list
                auc_user_dict[type] = auc_user_list
                logloss_dict[type] = logloss_list
                ndcg5_dict[type] = ndcg5_list
                ndcg10_dict[type] = ndcg10_list
                ndcg20_dict[type] = ndcg20_list
                hr5_dict[type] = hr5_list
                hr10_dict[type] = hr10_list
                hr20_dict[type] = hr20_list
                rate_dict[type] = rate_list
            else:
                auc_dict[type] = np.average(auc_list)
                auc_user_dict[type] = np.average(auc_user_list)
                logloss_dict[type] = np.average(logloss_list)
                ndcg5_dict[type] = np.average(ndcg5_list)
                ndcg10_dict[type] = np.average(ndcg10_list)
                ndcg20_dict[type] = np.average(ndcg20_list)
                hr5_dict[type] = np.average(hr5_list)
                hr10_dict[type] = np.average(hr10_list)
                hr20_dict[type] = np.average(hr20_list)
                rate_dict[type] = np.average(rate_list)

        for metric_name, metric_dict in zip(["auc", "auc_user", "logloss", "ndcg5", "ndcg10",
                                             "ndcg20", "hr5", "hr10", "hr20"],
                                            [auc_dict, auc_user_dict, logloss_dict, ndcg5_dict, ndcg10_dict,
                                             ndcg20_dict, hr5_dict, hr10_dict, hr20_dict]):
        # for metric_name, metric_dict in zip(["auc", "auc_user", "ndcg5", "ndcg10",
        #                                      "ndcg20", "hr5", "hr10", "hr20"],
        #                                     [auc_dict, auc_user_dict, ndcg5_dict, ndcg10_dict,
        #                                      ndcg20_dict, hr5_dict, hr10_dict, hr20_dict]):
            # max_value = 0
            value_list = []
            for type in TYPE_LIST:
                value_list.append(metric_dict[type][-1])
            max_value = max(value_list)
            for type in TYPE_LIST:
                diff = max_value - metric_dict[type][-1]
                for i in range(len(metric_dict[type])):
                    metric_dict[type][i] = metric_dict[type][i] + diff

        for metric_name, metric_dict in zip(["auc", "auc_user", "logloss", "ndcg5", "ndcg10",
                                             "ndcg20", "hr5", "hr10", "hr20"],
                                            [auc_dict, auc_user_dict, logloss_dict, ndcg5_dict, ndcg10_dict,
                                             ndcg20_dict, hr5_dict, hr10_dict, hr20_dict]):
        # for metric_name, metric_dict in zip(["auc", "auc_user", "ndcg5", "ndcg10",
        #                                      "ndcg20", "hr5", "hr10", "hr20"],
        #                                     [auc_dict, auc_user_dict, ndcg5_dict, ndcg10_dict,
        #                                      ndcg20_dict, hr5_dict, hr10_dict, hr20_dict]):
            # ax = plt.figure()
            # color_list
            # for type, color in zip(TYPE_LIST, ["red", "blue", "green"]):
            for type, color, legend_name, shape in zip(TYPE_LIST, ["green", "orange", "red", "blue"],
                                                       ["DUET (OC-SVM)", "DUET (LOF)", "DUET (Random)", "DUET (Ours)"],
                                                       ["*", "t"]):
                # TYPE_LIST = ["meta_random", "meta_ood", "meta_ood_ocsvm", "meta_ood_lof"]
                if type in ["meta_random", "meta_ood"]:
                    print(metric_name)
                    print(rate_dict[type])
                    print(metric_dict[type])
                    # plt.plot(rate_dict[type], metric_dict[type], label=type)
                    # plt.scatter(rate_dict[type], metric_dict[type], color=color, label=type)
                    # plt.plot(rate_dict[type], metric_dict[type], color=color, label=type)
                    plt.plot(rate_dict[type], metric_dict[type], color=color, label=legend_name)
                    # plt.plot(metric_dict[type])
                    # plt.savefig('performance.pdf', format='pdf', bbox_inches='tight')
                elif type in ["meta_ood_ocsvm", "meta_ood_lof"]:
                    # plt.scatter(rate_dict[type], metric_dict[type], color=color, label=type)
                    plt.scatter(rate_dict[type], metric_dict[type], color=color, marker=shape, label=type)
            plt.legend()
            plt.savefig(os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), '{}.png'.format(metric_name)), format='png', bbox_inches='tight')
            # plt.sca(ax)
            # plt.close()
            plt.clf()
            plt.cla()
