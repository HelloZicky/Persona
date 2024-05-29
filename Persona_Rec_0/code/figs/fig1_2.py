import os.path

import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import make_interp_spline

ROOT_FOLDER = "../checkpoint/SIGIR2023"
# DATASET_LIST = ["amazon_cds", "amazon_electronic", "amazon_beauty"]
DATASET_LIST = ["amazon_cds", "amazon_electronic"]
# TYPE_LIST = ["meta_ood"]
TYPE_LIST = ["meta_ood", "meta_ood_uncertainty5"]
MODEL_LIST = ["din", "sasrec", "gru4rec"]
log_filename = "fig1_list.txt"
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
# ymajorLocator = MultipleLocator(0.005)
# ymajorFormatter = FormatStrFormatter(':5s')
# def smooth_xy(lx, ly):
def smooth_xy(ly):
    """数据平滑处理

    :param lx: x轴数据，数组
    :param ly: y轴数据，数组
    :return: 平滑后的x、y轴数据，数组 [slx, sly]
    """
    x = np.array(np.arange(1, len(ly) + 1))
    # print(x)
    y = np.array(ly)
    x_smooth = np.linspace(min(x), max(x), (max(x) - min(x)) * 5)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    return [x_smooth, y_smooth]
    # return y_smooth

sample_num = 100
# group_num = 50
# group_num = 20
group_num = 25
group_num = 23
# group_num = 20
# start_index = 0
start_index = 10
start_index = 8
# start_index = 5
y_max = 0
y_min = 0
f = plt.figure()
for dataset in DATASET_LIST:
    print("{}{}{}".format("=" * 50, dataset, "=" * 50))
    for model in MODEL_LIST:
        print("{}{}{}".format("-" * 20, model, "-" * 20))

        for type in TYPE_LIST:
            auc_user_list = []
            auc_user_trigger_list = []
            mis_rec_list = []
            auc_diff_list = []
            file = os.path.join(ROOT_FOLDER, dataset + "_" + model, type, "fig1_list.txt")
            fig1_arr = np.loadtxt(file)
            for group_index in range(group_num):
                if group_index < start_index:
                    continue
                auc_user_item = np.mean(fig1_arr[group_index * sample_num: (group_index + 1) * sample_num, 0])
                auc_user_trigger_item = np.mean(fig1_arr[group_index * sample_num: (group_index + 1) * sample_num, 1])
                auc_user_list.append(auc_user_item)
                auc_user_trigger_list.append(auc_user_trigger_item)
                mis_rec_list.append(np.mean(fig1_arr[group_index * sample_num: (group_index + 1) * sample_num, 2]))
                # auc_diff_list.append((auc_user_item - auc_user_trigger_item) + 0.9)
                auc_diff_list.append((auc_user_item - auc_user_trigger_item))
            # auc_user = fig1_arr[:, 0]
            # auc_user_trigger = fig1_arr[:, 1]
            # mis_rec = fig1_arr[:, 2]
            auc_user = np.array(auc_user_list)
            auc_user_trigger = np.array(auc_user_trigger_list)
            mis_rec = np.array(mis_rec_list)
            # auc_diff = np.array(auc_diff_list)

            # print(fig1_arr[:, 0].shape)
            print(auc_user.shape)
            print(auc_user_trigger.shape)
            print(mis_rec.shape)
            # print(auc_diff.shape)

            # y_max = max(auc_user_list + auc_user_trigger_list + mis_rec_list + auc_diff_list)
            # y_min = min(auc_user_list + auc_user_trigger_list + mis_rec_list + auc_diff_list)
            y_max = max(auc_user_list + auc_user_trigger_list + mis_rec_list)
            y_min = min(auc_user_list + auc_user_trigger_list + mis_rec_list)

            auc_user, auc_user_trigger, mis_rec,  = \
                smooth_xy(auc_user[:group_num]), smooth_xy(auc_user_trigger[:group_num]), \
                smooth_xy(mis_rec[:group_num])

            # auc_user, auc_user_trigger, mis_rec, = \
            #     auc_user[:group_num], auc_user_trigger[:group_num], \
            #     mis_rec[:group_num]

            # plt.plot(auc_user[0], auc_user[1], color="r", label="AUC", alpha=0.5)
            # plt.plot(auc_user_trigger[0], auc_user_trigger[1], color="g", label="AUC (w/o request)", alpha=0.5)
            # plt.plot(mis_rec[0], mis_rec[1], color="b", label="MRS", alpha=0.5)
            # plt.plot(auc_diff[0], auc_diff[1], color="orange", label="AUC difference", alpha=0.5)
            plt.plot(auc_user[0], auc_user[1], color="r", alpha=0.5)
            plt.plot(auc_user_trigger[0], auc_user_trigger[1], color="g", alpha=0.5)
            plt.plot(mis_rec[0], mis_rec[1], color="b", alpha=0.5)
            # plt.plot(auc_diff[0], auc_diff[1], color="orange", alpha=0.5)
            # plt.legend()
            # import matplotlib.ticker as ticker
            # plt.locator_params(ticker.MultipleLocator(1))
            # plt.xticks(np.arange(0, group_num - start_index))
            # plt.xticks(np.arange(0, group_num - start_index + 1))
            f.set_figheight(3.5)
            plt.xticks(np.arange(1, group_num - start_index + 1))
            # plt.yticks(np.arange(0.7, 0.95, 0.05))
            # plt.yticks(np.arange(y_min - 0.05, y_max + 0.03, 0.02))
            # plt.yticks(np.arange(round((y_min - 0.02) * 1000 // 5 / 200, 2), round((y_max + 0.02) * 1000 // 5 / 200, 2), 0.01))

            # print(y_max)
            # y_max = "{:4f}".format(y_max)
            # y_min = "{:4f}".format(y_min)

            # print(round((y_max - 0.02), 4))
            # print(y_min)
            # print(round((y_min - 0.02), 4))
            # plt.yticks(np.arange(round((y_min - 0.02), 2), round((y_max + 0.02), 2), 0.005))
            # plt.yticks(np.arange(round((y_min - 0.02) * 1000 // 5 / 200, 3), round((y_max + 0.02) * 1000 // 5 / 200, 3), 0.01))
            # plt.yticks(np.arange(round((y_min - 0.02) * 200 // 1 / 200, 3), round((y_max + 0.02) * 200 // 1 / 200, 3), 0.01))

            # plt.yticks(np.arange((y_min - 0.02) * 200 // 1 / 200, (y_max + 0.02) * 200 // 1 / 200, 0.01))
            from decimal import *
            getcontext().prec = 3
            # a = Decimal(1) / Decimal(7)
            # ymajorFormatter = FormatStrFormatter('%1.5f')
            # ymajorLocator = MultipleLocator(0.005)
            plt.hlines(np.mean(mis_rec[1]), 1, group_num - start_index, linestyle='-.', colors="grey")
            # list_a = [Decimal(i) for i in np.arange((y_min - 0.02) * 200 // 1 / 200, (y_max + 0.02) * 200 // 1 / 200, 0.01)]

            # while True:
            #     y_min_temp = (y_min - 0.02) * 200 // 1
            #     y_max_temp = (y_max + 0.02) * 200 // 1
            #     # print("-" * 50)
            #     # print(y_max_temp)
            #     # print(y_min_temp)
            #     if y_min_temp % 2 != 0 or y_max_temp % 2 != 0:
            #         break
            #     if y_min_temp % 2 == 0:
            #         y_min = y_min - 0.005
            #         # print(y_min)
            #     if y_max_temp % 2 == 0:
            #         y_max = y_max + 0.005
            # list_a = np.arange((y_min - 0.02) * 200 // 1 / 200, (y_max + 0.02) * 200 // 1 / 200, 0.020)
            # list_a = np.arange((y_min - 0.02) * 1000 // 5 / 200, (y_max + 0.02) * 1000 // 5 / 200, 0.020)
            list_a = np.arange((y_min - 0.02) * 1000 // 10 / 100, (y_max + 0.02) * 1000 // 10 / 100, 0.020)
            plt.yticks(list_a)

            plt.xlim(0, group_num - start_index + 0.5)
            plt.xlabel("Group Number", fontsize=14)
            plt.ylabel("                    AUC", fontsize=14)
            plt.savefig(os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), '{}'.format(type), '{}_{}.png'.format("fig1", type)), format='png', bbox_inches='tight')
            plt.savefig(os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), '{}'.format(type), '{}_{}.pdf'.format("fig1", type)), format='pdf', bbox_inches='tight')

            plt.clf()
            plt.cla()

            auc_diff = np.array(auc_diff_list)
            print(auc_diff.shape)

            y_max = max(auc_diff)
            y_min = min(auc_diff)
            auc_diff = smooth_xy(auc_diff[:group_num])
            # auc_diff = auc_diff[:group_num]
            # f.set_figheight(0.8)
            f.set_figheight(1)
            plt.ylim(-0.015, 0.015)
            # plt.plot(auc_diff[0], auc_diff[1], color="orange", alpha=0.5)
            # plt.yticks(np.arange(round((y_min - 0.02) * 1000 // 5 / 200, 2), round((y_max + 0.02) * 1000 // 5 / 200, 2), 0.004))
            plt.plot(auc_diff[0], auc_diff[1], color="orange", alpha=0.5)

            # plt.yticks(np.arange(round((y_min - 0.02) * 1000 // 5 / 200, 2), round((y_max + 0.03) * 1000 // 5 / 200, 2), 0.005))
            # plt.yticks(np.arange(round((y_min - 0.02) * 1000 // 5 / 200, 2), round((y_max + 0.03) * 1000 // 5 / 200, 2), 0.005))
            # plt.yticks(np.arange(round((y_min - 0.001) * 1000 // 5 / 200, 2), round((y_max + 0.001) * 1000 // 5 / 200, 2), 0.01))
            # plt.plot(auc_diff[0] * 100, auc_diff[1] * 100, color="orange", alpha=0.5)
            # plt.yticks(np.arange(round((y_min - 0.02) * 1000 // 5 / 200, 2) * 100, round((y_max + 0.02) * 1000 // 5 / 200, 2) * 100, 0.004 * 100))
            # plt.xticks(np.arange(0, group_num - start_index))

            # plt.xticks(np.arange(1, group_num - start_index + 1))
            plt.xticks([])
            plt.hlines(0, 1, group_num - start_index, linestyle='--', colors="skyblue")
            # plt.xlabel("Group Number", fontsize=14)
            # plt.ylabel("AUC", fontsize=14)
            plt.savefig(os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), '{}'.format(type), '{}_{}_prifit.png'.format("fig1", type)), format='png', bbox_inches='tight')
            plt.savefig(os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), '{}'.format(type), '{}_{}_prifit.pdf'.format("fig1", type)), format='pdf', bbox_inches='tight')

            plt.clf()
            plt.cla()

            f.set_figheight(4.5)
            plt.plot(auc_user[0], auc_user[1], color="r", alpha=0.5)
            plt.plot(auc_user_trigger[0], auc_user_trigger[1], color="g", alpha=0.5)
            plt.plot(mis_rec[0], mis_rec[1], color="b", alpha=0.5)
            plt.plot(auc_diff[0], auc_diff[1], color="orange", alpha=0.5)
            # plt.xticks(np.arange(0, group_num - start_index))
            plt.xticks(np.arange(1, group_num - start_index + 1))
            # plt.hlines(metric_dict[type], 0, 100, linestyle='--', colors="skyblue", label=legend_name)
            plt.xlabel("Group Number", fontsize=14)
            plt.ylabel("AUC", fontsize=14)
            plt.savefig(os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), '{}'.format(type), '{}_{}_overall.png'.format("fig1", type)), format='png', bbox_inches='tight')
            plt.savefig(os.path.join(ROOT_FOLDER, "{}_{}".format(dataset, model), '{}'.format(type), '{}_{}_overall.pdf'.format("fig1", type)), format='pdf', bbox_inches='tight')

            plt.clf()
            plt.cla()
