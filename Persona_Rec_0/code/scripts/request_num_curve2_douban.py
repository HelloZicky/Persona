import math
import os
import numpy
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# ROOT_FOLDER = "../checkpoint/NIPS2022"
# ROOT_FOLDER = "../checkpoint/WWW2023"
ROOT_FOLDER = "../checkpoint/SIGIR2023"
# ROOT_FOLDER_ = "../checkpoint/SIGIR2023_backup"
ROOT_FOLDER_ = "../checkpoint/SIGIR2023"
# ROOT_FOLDER = "../checkpoint/SIGIR2023_backup"
# DATASET_LIST = ["amazon_cds", "amazon_tv", "amazon_electronic", "amazon_beauty", "amazon_clothing", "amazon_books", "amazon_sports", "movielens", "movielens_100k", ]
# DATASET_LIST = ["amazon_cds", "amazon_electronic", "amazon_beauty", "movielens"]
# DATASET_LIST = ["amazon_cds", "amazon_electronic", "amazon_beauty"]
# DATASET_LIST = ["amazon_cds", "amazon_electronic"]
DATASET_LIST = ["douban_book", "douban_music"]
# TYPE_LIST = ["base", "base_finetune", "meta", "meta_gru", "meta_random", "meta_ood_ocsvm", "meta_ood_lof", "meta_ood_if", "meta_ood", "meta_ood_gru", "meta_ood_uncertainty"]
# TYPE_LIST = ["meta_random", "meta_ood", "meta_ood_uncertainty"]
# TYPE_LIST = ["meta_random", "meta_ood"]
# TYPE_LIST = ["meta_random", "meta_ood", "meta_ood_ocsvm", "meta_ood_lof"]
# TYPE_LIST = ["meta_ood_ocsvm", "meta_ood_lof", "meta_random", "meta_ood", "meta_ood_uncertainty"]
# TYPE_LIST = ["meta_ood_ocsvm", "meta_ood_lof", "meta_random", "meta_ood", "meta_ood_uncertainty6"]
# TYPE_LIST = ["meta_ood_ocsvm", "meta_ood_lof", "meta_random", "meta_ood", "meta_ood_uncertainty", "meta_ood_uncertainty6"]
# TYPE_LIST = ["meta_ood_ocsvm", "meta_ood_lof", "meta_random", "meta_ood", "meta_ood_uncertainty", "meta_ood_uncertainty5", "meta_ood_uncertainty6"]
TYPE_LIST = ["meta_ood_ocsvm", "meta_ood_lof", "meta_random", "meta_ood", "meta_ood_uncertainty5"]
# TYPE_LIST_overall = ["meta", "meta_ood_ocsvm", "meta_ood_lof", "meta_random", "meta_ood", "meta_ood_uncertainty", "meta_ood_uncertainty6"]
# TYPE_LIST_reverse = ["meta_random", "meta_ood", "meta_ood_ocsvm", "meta_ood_lof"]
# TYPE_LIST_reverse = ["meta_random", "meta_ood", "meta_ood_ocsvm", "meta_ood_lof", "meta_ood_uncertainty", "meta_ood_uncertainty5", "meta_ood_uncertainty6"]
# TYPE_LIST_reverse = ["meta_random", "meta_ood", "meta_ood_ocsvm", "meta_ood_lof", "meta_ood_uncertainty5"]
TYPE_LIST_reverse = ["meta_random", "meta_ood", "meta_ood_ocsvm", "meta_ood_lof", "meta_ood_uncertainty5", "meta_ood_uncertainty5_2"]
# DATASET_LIST = ["movielens", "movielens_100k"]
# DATASET_LIST = ["amazon_beauty"]
# MODEL_LIST = ["din", "gru4rec", "sasrec"]
MODEL_LIST = ["din", "sasrec", "gru4rec"]

log_filename = "test.txt"
log_filename_ = "log_ood.txt"

result_file = os.path.join(ROOT_FOLDER, "request_curve.txt")
fig_file = os.path.join(ROOT_FOLDER, "request_curve.png")

# interpolation = True
interpolation = False
def closest_value(list_a, k):
    return list_a[min(range(len(list_a)), key=lambda i: abs(list_a[i]-k))]


def smooth_xy(lx, ly):
    """数据平滑处理

    :param lx: x轴数据，数组
    :param ly: y轴数据，数组
    :return: 平滑后的x、y轴数据，数组 [slx, sly]
    """
    x = np.array(lx)
    y = np.array(ly)
    x_smooth = np.linspace(min(x), max(x), 5)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    return x_smooth, y_smooth

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
        for type in TYPE_LIST + ["meta"]:
            print("-" * 50)
            print("type ", type)
            # if type in ["meta_random", "meta_ood", "meta_ood_uncertainty6"]:
            if type in ["meta_random", "meta_ood", "meta_ood_uncertainty", "meta_ood_uncertainty5", "meta_ood_uncertainty6", "meta_ood_ocsvm", "meta_ood_lof"]:
                log_file = os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), type, log_filename)
            # elif type in ["meta_ood_ocsvm", "meta_ood_lof"]:
            # elif type in ["meta_ood_ocsvm", "meta_ood_lof", "meta"]:
            elif type in ["meta"]:
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
                    # if type in ["meta_ood_ocsvm", "meta_ood_lof"]:
                    #     rate = float(line.strip("\n").split(",")[-4].split("=")[-1]) / float(line.strip("\n").split(",")[-3].split("=")[-1])
                    # elif type == "meta":
                    #     rate = 1.0
                    # else:
                    #     rate = float(line.strip("\n").split(",")[-1].split("=")[-1])

                    if type == "meta":
                        rate = 1.0
                    else:
                        rate = float(line.strip("\n").split(",")[-1].split("=")[-1])
                    print(line)
                    print(rate)

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
                    # if type in ["meta_ood_ocsvm", "meta_ood_lof"] + ["meta"]:
                    if type in ["meta"]:
                        if index % 10 == 0:
                            # print(auc_list)
                            auc_dict[type].append(max(auc_list))
                            print(auc_dict[type])
                            auc_user_dict[type].append(max(auc_user_list))
                            logloss_dict[type].append(max(logloss_list))
                            ndcg5_dict[type].append(max(ndcg5_list))
                            ndcg10_dict[type].append(max(ndcg10_list))
                            ndcg20_dict[type].append(max(ndcg20_list))
                            hr5_dict[type].append(max(hr5_list))
                            hr10_dict[type].append(max(hr10_list))
                            hr20_dict[type].append(max(hr20_list))
                            rate_dict[type].append(rate_list[auc_user_dict[type].index(max(auc_user_dict[type]))])
                            # print(rate_dict[type])

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

                        # if index > 50:
                        if index > 10:
                            break

                    if rate == float(100):
                        break
            # print("type ", type)
            # if type in ["meta_random", "meta_ood", "meta_ood_uncertainty", "meta_ood_uncertainty5", "meta_ood_uncertainty6"]:
            if type in ["meta_random", "meta_ood", "meta_ood_uncertainty", "meta_ood_uncertainty5", "meta_ood_uncertainty6", "meta_ood_lof", "meta_ood_ocsvm"]:
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
                auc_dict[type] = np.average(auc_dict[type])
                # print("-" * 50)
                # print(auc_dict[type])
                # print(auc_list)
                # print(np.average(auc_list))
                # print(np.average(auc_list, axis=-1))
                # print(np.array(auc_dict[type]).shape)
                auc_user_dict[type] = np.average(auc_user_dict[type])
                logloss_dict[type] = np.average(logloss_dict[type])
                ndcg5_dict[type] = np.average(ndcg5_dict[type])
                ndcg10_dict[type] = np.average(ndcg10_dict[type])
                ndcg20_dict[type] = np.average(ndcg20_dict[type])
                hr5_dict[type] = np.average(hr5_dict[type])
                hr10_dict[type] = np.average(hr10_dict[type])
                hr20_dict[type] = np.average(hr20_dict[type])
                rate_dict[type] = np.average(rate_dict[type])
                print(rate_dict[type])

        metric_max_dict = defaultdict(list)
        metric_min_dict = defaultdict(list)
        value_dict = defaultdict(list)

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
            # print("=" * 50)
            # print(type)
            # print(metric_name)
            # print(metric_dict[type])
            for type in TYPE_LIST:
                print(type)
                print(metric_dict[type])
                # if type in ["meta_ood_ocsvm", "meta_ood_lof"]:
                #     continue
                value_list.append(metric_dict[type][-1])
                # rate_dict[type].append(rate_list[auc_user_dict[type].index(max(auc_user_dict[type]))])
                # value_list_lof.append(metric_dict[type][-1])
                # value_list_lof.append(metric_dict[type][rate_dict[type]])
                # value_list_ocsvm.append(metric_dict[type][-1])
                for i in range(len(metric_dict[type])):
                    value_dict[rate_dict[type][i]].append(metric_dict[type][i])

            max_value = max(value_list)
            min_value = min(value_list)
            diff = 0

            # TYPE_LIST_ = TYPE_LIST.reverse()
            diff_point = max_value - min_value

            # print(" == " * 50)
            # print('dataset ', dataset)
            # print('model ', model)
            # print('metric_name ', metric_name)
            # print('metric_dict["meta"] ', metric_dict["meta"])
            # print('max_value ', max_value)
            # print('value_list ', value_list)

            # diff_temp = max_value - metric_dict["meta"]
        
            # diff_temp = abs(max_value - metric_dict["meta"])
            # for type in TYPE_LIST_reverse:
            #     if type in ["meta_ood_ocsvm", "meta_ood_lof"]:
            #         # metric_dict[type] = metric_dict[type] + diff / 2
            #         # metric_dict[type] = metric_dict[type] + diff
            #         # metric_dict[type] = metric_dict[type] + diff_temp
            #         continue
            #
            #     # if type in ["meta_ood_ocsvm", "meta_ood_lof"]:
            #     #     # continue
            #     #     # print(value_dict)
            #     #     # print(rate_dict[type])
            #     #     # print((round(rate_dict[type][-1] / 10, 0) * 10))
            #     #     # print(int(round(rate_dict[type][-1] / 10, 0) * 10))
            #     #     # index = value_dict[(round(rate_dict[type][-1] / 10, 0) * 10)]
            #     #     index = closest_value(list(value_dict.keys()), rate_dict[type])
            #     #
            #     #     diff_point = max(value_dict[index]) - min(value_dict[index])
            #     #     if dataset == "amazon_cds":
            #     #         print("#" * 10)
            #     #         # print(diff)
            #     #         print(diff_point)
            #     #     # metric_dict[type] = metric_dict[type] + diff
            #     #     metric_dict[type] = metric_dict[type] + diff_point
            #     #     # metric_dict[type] = metric_dict[type] - diff_point
            #     #
            #     #     print("+" * 100)
            #     #     print(max(value_dict[index]))
            #     #     print(min(value_dict[index]))
            #     #     print(max(value_dict[index]) - min(value_dict[index]))
            #     #     print(metric_dict[type])
            #     #     print(metric_dict[type] + diff_point)
            #     #     print("-" * 100)
            #     #     continue
            #
            #     diff = max_value - metric_dict[type][-1]
            #     # print("+" * 50)
            #     # print(type)
            #     # print(metric_dict[type][-1] + diff)
            #     for i in range(len(metric_dict[type])):
            #         metric_dict[type][i] = metric_dict[type][i] + diff

            metric_dict["meta"] = max_value

        for metric_name, metric_dict, ylabel in zip(["auc", "auc_user", "logloss", "ndcg5", "ndcg10",
                                             "ndcg20", "hr5", "hr10", "hr20"],
                                            [auc_dict, auc_user_dict, logloss_dict, ndcg5_dict, ndcg10_dict,
                                             ndcg20_dict, hr5_dict, hr10_dict, hr20_dict],
                                            ["AUC", "UAUC", "LogLoss", "NDCG@5", "NDCG@10",
                                             "NDCG@20", "HR@5", "HR@10", "HR@20"],):
        # for metric_name, metric_dict in zip(["auc", "auc_user", "ndcg5", "ndcg10",
        #                                      "ndcg20", "hr5", "hr10", "hr20"],
        #                                     [auc_dict, auc_user_dict, ndcg5_dict, ndcg10_dict,
        #                                      ndcg20_dict, hr5_dict, hr10_dict, hr20_dict]):
            # ax = plt.figure()
            # color_list
            # for type, color in zip(TYPE_LIST, ["red", "blue", "green"]):
            # min_index = 0 + 3
            # min_index = 0 + 4
            min_index = 0
            # min_index = 0 + 2
            # min_index = 0
            # max_index = 16 - 6
            # max_index = 16 - 3
            # max_index = 16
            max_index = 17
            # for type, color, legend_name, shape in zip(TYPE_LIST + ["meta"], ["green", "orange", "red", "blue", "grey", "black", "orange"],
            #                                            ["DUET (OC-SVM)", "DUET (LOF)", "DUET (Random)", "DUET (Ours 1)", "DUET (Ours 2)", "DUET (Ours 3)", "DUET"],
            #                                            ["*", "o", "s", "x", ".", "_", "_"]):
            # for type, color, legend_name, shape in zip(TYPE_LIST + ["meta"], ["green", "orange", "red", "blue", "grey", "black", "orange", "skyblue"],
            #                                            ["DUET (OC-SVM)", "DUET (LOF)", "DUET (Random)", "MR-DUET (w/o DM)", "MR-DUET (DM1)", "MR-DUET (DM5)", "MR-DUET (DM6)", "DUET"],
            #                                            ["*", "o", "s", "x", ".", "_", "_", "_"]):
            metric_list = []
            rate_list = []
            for type, color, legend_name, shape in zip(TYPE_LIST + ["meta"], ["green", "orange", "red", "blue", "black", "skyblue"],
                                                   # ["DUET (OC-SVM)", "DUET (LOF)", "DUET (Random)", "MR-DUET (w/o DM)", "MR-DUET (DM5)", "DUET"],
                                                   ["DUET (OC-SVM)", "DUET (LOF)", "DUET (Random)", "MR-DUET (w/o DM)", "MR-DUET", "DUET"],
                                                   ["*", "o", "s", "x", ".", "_", "_", "_"]):
                # TYPE_LIST = ["meta_random", "meta_ood", "meta_ood_ocsvm", "meta_ood_lof"]
                # print(TYPE_LIST + ["meta"])

                # print("*" * 50)
                # print("type ", type)
                # if type in ["meta_ood_uncertainty", "meta_ood_uncertainty6"]:
                #     continue

                if type in ["meta_random", "meta_ood", "meta_ood_uncertainty", "meta_ood_uncertainty5", "meta_ood_uncertainty6", "meta_ood_ocsvm", "meta_ood_lof"]:
                # if type in ["meta_random", "meta_ood", "meta_ood_uncertainty", "meta_ood_uncertainty5", "meta_ood_uncertainty6"]:
                    # print(1)
                    # print(type)
                    # print(metric_name)
                    # print(rate_dict[type])
                    # print(metric_dict[type])

                    # plt.plot(rate_dict[type], metric_dict[type], label=type)
                    # plt.scatter(rate_dict[type], metric_dict[type], color=color, label=type)
                    # plt.plot(rate_dict[type], metric_dict[type], color=color, label=type)
                    if interpolation:
                        rate_dict_temp, metric_dict_temp = smooth_xy(rate_dict[type][min_index:max_index], metric_dict[type][min_index:max_index])

                    else:
                        rate_dict_temp, metric_dict_temp = rate_dict[type][min_index:max_index], metric_dict[type][min_index:max_index]

                    metric_list.extend(metric_dict_temp)
                    plt.plot(rate_dict_temp, metric_dict_temp, color=color, label=legend_name, alpha=0.7)

                    # plt.plot(metric_dict[type])
                    # plt.savefig('performance.pdf', format='pdf', bbox_inches='tight')
                # elif type in ["meta_ood_ocsvm", "meta_ood_lof"]:
                #     # print(2)
                #     # print(type)
                #     # print(metric_name)
                #     # print(rate_dict[type])
                #     # print(metric_dict[type])
                #
                #     # plt.scatter(rate_dict[type], metric_dict[type], color=color, label=type)
                #     # print(type)
                #     # print(rate_dict[type])
                #     # print(metric_dict[type])
                #     # plt.scatter(rate_dict[type] * 100, metric_dict[type], color=color, marker=shape, label=legend_name)
                #     plt.scatter(rate_dict[type][-1], metric_dict[type][-1], color=color, marker=shape, label=legend_name)
                #     # continue

                elif type == "meta":
                # if type == "meta":
                #     print(3)
                #     print("|||| ", metric_dict[type])
                #     print(metric_dict[type])
                    metric_list.append(metric_dict[type])
                    plt.hlines(metric_dict[type], 0, 100, linestyle='--', colors=color, label=legend_name)

            y_max = max(metric_list)
            y_min = min(metric_list)
            # plt.legend()
            plt.xlabel("Request Frequency", fontsize=16)
            plt.ylabel(ylabel, fontsize=16)

            # plt.yticks(np.arange(round((y_min - 0.005) * 1000 // 5 / 200, 3), round((y_max + 0.005) * 1000 // 5 / 200, 3), 0.005))
            # plt.yticks(np.arange(round((y_min) * 1000 // 5 / 200, 3), round((y_max) * 1000 // 5 / 200, 3), 0.005))
            plt.yticks(np.arange(round((y_min), 3), round((y_max), 3), 0.001))
            # plt.xlim(0, group_num - start_index + 0.5)

            # plt.xlabel("Model", fontsize=16)
            # plt.ylabel('Accuracy(%)', fontsize=16)
            plt.tick_params(labelsize=12)

            plt.savefig(os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), '{}.png'.format(metric_name)), format='png', bbox_inches='tight')
            plt.savefig(os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), '{}.pdf'.format(metric_name)), format='pdf', bbox_inches='tight')
            # plt.sca(ax)
            # plt.close()
            plt.clf()
            plt.cla()
