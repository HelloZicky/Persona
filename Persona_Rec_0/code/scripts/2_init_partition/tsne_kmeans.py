import os.path
import sys
from colorsys import hls_to_rgb

import numpy as np
# from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from func_kmeans import func_kmeans
import random
from matplotlib.colors import cnames
# torch.set_num_threads(10)
# torch.set_num_threads(8)
# torch.set_num_threads(4)
torch.set_num_threads(3)


def get_color_list():
    color_list = [
        # 'aqua',
        #           'black',
                  'blue',
                  'brown',
                  # 'darkcyan',
                  'darkgreen',
                  'darkmagenta',
                  'darkorchid',
                  'darkred',
                  'darkslategray',
                  'darkviolet',
                  'deeppink',
                  'fuchsia',
                  'indigo',
                  'lime',
                  "orange",
                  "red",
                  "black",
                  "green",
                  "maroon",
                  "midnightblue",
                  'mediumblue',
                  'mediumorchid',
                  'mediumpurple',
                  'mediumseagreen',
                  'mediumslateblue',
                  'mediumspringgreen',
                  'mediumturquoise',
                  'mediumvioletred',
                  'midnightblue',
                  "dodgerblue",
                  "chocolate",
                  "goldenrod",
                  'magenta',
                  'maroon',
                  'navy',
                  'olive',
                  'olivedrab',
                  'orangered',
                  'orchid',
                  'palegoldenrod',
                  'palegreen',
                  'paleturquoise',
                  'palevioletred',
                  'peru',
                  'pink',
                  'plum',
                  'powderblue',
                  'purple',
                  'red',
                  'rosybrown',
                  'royalblue',
                  'saddlebrown',
                  'salmon',
                  'sandybrown',
                  'seagreen',
                  "yellow"
                  # 'cyan'
                  ]
    # return color_list * 100 + ["cyan"]
    return color_list


def get_distinct_colors(n):

    colors = []

    # for i in np.arange(0., 360., 360. / n):
    #     h = i / 360.
    #     l = (50 + np.random.rand() * 10) / 100.
    #     s = (90 + np.random.rand() * 10) / 100.
    #     colors.append(hls_to_rgb(h, l, s))

    for i in np.arange(0., 255., 255. / n):
        h = i / 255.
        l = (50 + np.random.rand() * 10) / 255.
        s = (90 + np.random.rand() * 10) / 255.
        colors.append(hls_to_rgb(h, l, s))

    return colors


def plot_embedding(tsne_embedding, db_labels=None, title=""):
    # color_list = ["black", "blue", "red", "green", "grey", "purple", "orange", "darkblue", "darkgoldenrod", "cyan"]

    # color_list = get_distinct_colors(len(db_labels))
    color_list = list(set(get_color_list()))
    # color_list = get_distinct_colors()

    # color_list = ["black", "blue", "red", "green", "grey", "purple"]
    x_min, x_max = np.min(tsne_embedding, 0), np.max(tsne_embedding, 0)
    data = (tsne_embedding - x_min) / (x_max - x_min)
    # data = tsne_embedding
    fig = plt.figure()
    ax = plt.subplot(111)

    print(data.shape)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    if db_labels is None:

        return

    else:
        # for index, point in enumerate(data[:-1]):
        for index, point in enumerate(data):
            # if db_labels[index] == -1:
            #     ax.scatter(data[index][0], data[index][1], s=3,
            #                # color=color_list[db_labels[index]],
            #                color="cyan",
            #                label=db_labels[index], alpha=0.7)
            #     continue
            # ax.scatter(data[index][0], data[index][1], s=3,
            #            # color=color_list[db_labels[index]],
            #            c=cnames[color_list[db_labels[index]]],
            #            # c=db_labels[index] * 10,
            #            label=db_labels[index],
            #            # []
            #            # alpha=0.7
            #            )
            if db_labels[index] == -1:
                # ax.scatter(data[index][0], data[index][1], s=3,
                #            c=cnames[color_list[db_labels[index]]],
                #            label=db_labels[index],
                #            alpha=0.3
                #            )
                continue
            else:
                # ax.scatter(data[index][0], data[index][1], s=5,
                #            # c=cnames[color_list[db_labels[index]]],
                #            # color=rgba(db_labels[index], db_labels[index], db_labels[index]),
                #            c=np.array(
                #                [
                #                    # (max(db_labels) - float(db_labels[index])) / max(db_labels),
                #                    # float(db_labels[index] / max(db_labels)),
                #                    # float(db_labels[index] / max(db_labels)),
                #                    # float(db_labels[index] / max(db_labels)),
                #                    float(db_labels[index] / 600),
                #                    float(db_labels[index] / 600),
                #                    float(db_labels[index] / 600),
                #                ]
                #            ).reshape(1, -1),
                #            label=db_labels[index],
                #            alpha=0.7
                #            )
                # cycle_num = db_labels[index] // len(color_list)
                ax.scatter(data[index][0], data[index][1], s=5,
                           # c=cnames[color_list[db_labels[index]]],
                           c=cnames[color_list[db_labels[index] % len(color_list)]],
                           label=db_labels[index],
                           alpha=max([0.2, 1 - (db_labels[index] / len(color_list)) / 20])
                           )

    return fig


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Jenkins pipline parameters')

    # parser.add_argument('--eps', type=float, default=0.01, help='uuid value')
    # parser.add_argument('--min_samples', type=int, default=100, help='uuid value')
    # parser.add_argument('--class_num', type=int, default=100, help='uuid value')
    parser.add_argument('--class_num', type=int, default=10, help='uuid value')
    parser.add_argument('--dataset', type=str, default="amazon_beauty", help='uuid value')
    parser.add_argument('--model', type=str, default="din", help='uuid value')
    args = parser.parse_args()
    # print("eps={}, min_samples={}".format(args.eps, args.min_samples))
    print("class_num={}".format(args.class_num))

    # chechpoint_folder = "checkpoint_0828/NIPS2022/"
    # chechpoint_folder = "../checkpoint/NIPS2022/"
    # chechpoint_folder = "../checkpoint/SIGIR2023/"
    chechpoint_folder = "../../checkpoint/NIPS2023/"
    # model_folder_list = ["alipay_din", "alipay_gru4rec", "alipay_sasrec"]
    # model_folder_list = ["amazon_cds_din", "amazon_cds_gru4rec", "amazon_cds_sasrec"]
    # model_folder_list = ["amazon_cds_din"]
    # model_folder_list = ["amazon_beauty_din"]
    model_folder_list = ["{}_{}".format(args.dataset, args.model)]
    # model_folder_list = ["alipay_gru4rec", "alipay_sasrec"]
    # type_folder = "base"
    type_folder = "meta_grad_gru"
    # pt_folder_list = ["hist_embed.pt", "target_embed.pt"]
    # pt_folder_list = ["hist_embed.pt"]
    # pt_folder_list = ["target_embed.pt"]
    # dimension = 32
    # sample_num = 5000
    sample_num = None
    # random.seed(628)
    # random.seed(1125)
    # sample_num = None
    # x_list = []
    # x_train_list = torch.nn.ModuleList([])
    # x_test_list = torch.nn.ModuleList([])
    x_train_list = []
    x_test_list = []
    train_sample_num = 0
    test_sample_num = 0
    for model in model_folder_list:
        pt_folder_list = [pt for pt in os.listdir(os.path.join(chechpoint_folder, model, type_folder)) if pt.split(".")[-1] == "pt"]
        for index, pt in enumerate(pt_folder_list):
            mode = pt.split("_")[0]

            print("{} {}".format(model, pt))
            pt_path = os.path.join(chechpoint_folder, model, type_folder, pt)
            x = torch.load(pt_path)
            # x = np.array(x).flatten()
            if sample_num is not None:
                x = random.sample(x, sample_num)
            print(len(x))

            # x_list.extend(x)
            if mode == "train":
                # x_train_list.extend(x)
                x_train_list.append(torch.stack(x))
                train_sample_num = torch.stack(x).size()[0]
                # x_train_list.extend(torch.stack(x))
            elif mode == "test":
                # x_test_list.extend(x)
                x_test_list.append(torch.stack(x))
                test_sample_num = torch.stack(x).size()[0]
            # else:
            #     continue
            print(torch.stack(x).size())
            # x_list.append(x)
            # x_list.append(np.array(x))
        # x = torch.stack(x, dim=0)
        try:
            x_train = torch.stack(x_train_list, dim=2).view(train_sample_num, -1)
            x_test = torch.stack(x_test_list, dim=2).view(test_sample_num, -1)
            print(x_train.size())
            # print(x_train.squeeze(2).size())
            # print(x_train.view(train_sample_num, -1).size())
            # print(x_train.squeeze(1).size())
            print(x_test.size())
    
            # x = torch.stack(x_list, dim=0)
            # x = torch.stack([x_train, x_test], dim=0)
            # x = torch.cat([x_train, x_test], dim=0)
            x = x_train
            # x = torch.stack(np.array(x_list), dim=0)
            # x = torch.stack(x_list, dim=1)
            # x = torch.stack(np.array(x_list))
            print(x.size())
        except RuntimeError:
            print(model)
            # print()
        # db_labels = func_kmeans(x, args.eps, args.min_samples)
        kmeans = func_kmeans(x, args.class_num)
        db_labels = kmeans.labels_
        # if max(db_labels) - min(db_labels) <= 4:
        #     print("---exit---")
        #     os._exit(0)
        print("class_num={}, len(db_labels)={}, max(db_labels)={}, min(db_labels)={},".format(
            args.class_num, len(db_labels), max(db_labels), min(db_labels)
        ))
        # with open('amazon_{}_{}_eps{}_ms{}.txt'.format(len(x), model, args.eps, args.min_samples), "w+") as writer:
        #     print("class_num={}, len(db_labels)={}, max(db_labels)={}, min(db_labels)={},".format(
        #         args.class_num, len(db_labels), max(db_labels), min(db_labels)
        #     ), file=writer)
        #     print(db_labels, file=writer)

        # with open('amazon_{}_{}_classnum{}.txt'.format(len(x), model, args.class_num), "w+") as writer:
        #     # print("class_num={}, len(db_labels)={}, max(db_labels)={}, min(db_labels)={},".format(
        #     #     args.class_num, len(db_labels), max(db_labels), min(db_labels)
        #     # ), file=writer)
        #     print(db_labels, file=writer)
        # np.savetxt('amazon_{}_{}_classnum{}.txt'.format(len(x), model, args.class_num), db_labels)
        # np.savetxt('amazon_{}_{}_classnum{}.txt'.format(len(x), model, args.class_num), db_labels, fmt='%d')

        # np.savetxt('amazon_{}_classnum_{}.txt'.format(model, args.class_num), db_labels, fmt='%d')
        # np.savetxt('amazon_{}_classnum_{}_train.txt'.format(model, args.class_num), db_labels[:train_sample_num], fmt='%d')
        # np.savetxt('amazon_{}_classnum_{}_test.txt'.format(model, args.class_num), db_labels[-test_sample_num:], fmt='%d')
        folder = os.path.join(args.dataset, args.model)
        if not os.path.exists(folder):
            os.makedirs(folder)
        # np.savetxt(os.path.join(folder, '{}_classnum_{}.txt'.format(model, args.class_num)), db_labels, fmt='%d')
        np.savetxt(os.path.join(folder, '{}_classnum_{}_train.txt'.format(model, args.class_num)), db_labels, fmt='%d')
        # test_label = []
        # for x_test_item in x_test:
        #     test_label.append(int(kmeans.predict(x_test_item)))
        # x_test = np.array(list(map(lambda x: x.astype('double'), x_test)))
        # x_test = np.array(x_test_list)
        # x_test = x_test.numpy().reshape(x_test.size()[0], 1)

        x_test = x_test.numpy()

        # print("-" * 5)
        # print(x_test.size())
        # test_label = kmeans.predict(x_test).reshape(-1, 1)
        test_label = kmeans.predict(x_test)
        print(test_label.shape)
        np.savetxt(os.path.join(folder, '{}_classnum_{}_test.txt'.format(model, args.class_num)), test_label, fmt='%d')
        # np.savetxt(os.path.join(folder, '{}_classnum_{}_test.txt'.format(model, args.class_num)), np.array(test_label), fmt='%d')
        # np.savetxt(os.path.join(folder, '{}_classnum_{}_test.txt'.format(model, args.class_num)), np.array(test_label))
        
        # np.savetxt(os.path.join(folder, '{}_classnum_{}_train.txt'.format(model, args.class_num)), db_labels[:train_sample_num], fmt='%d')
        # np.savetxt(os.path.join(folder, '{}_classnum_{}_test.txt'.format(model, args.class_num)), db_labels[-test_sample_num:], fmt='%d')
        print(kmeans.cluster_centers_)
        # np.savetxt(os.path.join(folder, '{}_classnum_{}_cluster_center.txt'.format(model, args.class_num)), kmeans.cluster_centers_, fmt='%d')
        np.savetxt(os.path.join(folder, '{}_classnum_{}_cluster_center.txt'.format(model, args.class_num)), kmeans.cluster_centers_)
        # print(x.size())
        # x = x[:, :x.size()[1] // 2]
        # x = x[:, :dimension]

        # print(x.size())
        # X = np.array([np.array(i.cpu()).reshape(-1) for i in x])

        # tsne
        # tsne = TSNE(n_components=2, init='pca', random_state=0)
        #
        # x_tsne = tsne.fit_transform(x)
        # print("after tsne")
        # print(x_tsne.shape)
        # x_tsne = np.array(x_tsne)
        # db_labels = np.array(db_labels)
        # print("x_tsne.shape {} \t db_labels.shape {}".format(x_tsne.shape, db_labels.shape))

        # x_tsne = x_tsne[db_labels != -1]
        # db_labels = db_labels[db_labels != -1]
        # print("after filter -1")
        # print("db_label\n", db_labels)
        # print("x_tsne.shape {} \t db_labels.shape {}".format(x_tsne.shape, db_labels.shape))

        # # print(x_tsne.size())
        # fig = plot_embedding(x_tsne, db_labels, "t-SNE embedding for convolutional layers")
        # # plt.savefig('alipay_{}.pdf'.format(model), format='pdf', bbox_inches='tight')
        # # plt.savefig('movielens_{}_{}_{}_{}.png'.format(len(x), dimension, model, pt.split(".")[0].split("_")[0]), format='png', bbox_inches='tight')
        # # plt.savefig('amazon_{}_{}_{}_{}.png'.format(len(x), dimension, model, pt.split(".")[0].split("_")[0]), format='png', bbox_inches='tight')
        # # plt.savefig('amazon_{}_{}_{}.png'.format(len(x), model, pt.split(".")[0].split("_")[0]), format='png', bbox_inches='tight')
        # # plt.savefig('amazon_{}_{}_eps{}_ms{}_{}.png'.format(len(x), model, args.eps, args.min_samples, len(x_tsne)),
        # #             format='png', bbox_inches='tight')
        # plt.savefig('amazon_{}_{}_classnum{}.png'.format(len(x), model, args.class_num),
        #             format='png', bbox_inches='tight')
        sys.exit(0)