import os.path

import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import torch


def func_kmeans(x, class_num=50):
    X = np.array([np.array(i.cpu()).reshape(-1) for i in x])
    # db = KMeans(n_clusters=class_num, random_state=0, n_init="auto").fit(X)
    db = KMeans(n_clusters=class_num, random_state=0).fit(X)
    # db = KMeans.fit_predict(X)

    # db = DBSCAN(eps=1, min_samples=20).fit(X)
    print("-" * 50)
    print(len(db.labels_))
    print(db.labels_)
    print("-" * 50)
    # print(db.size())
    # print(db)

    # return db.labels_
    return db


# def func_dbscan(X):
#     # chechpoint_folder = "checkpoint_0828/NIPS2022/"
#     chechpoint_folder = "checkpoint/NIPS2022/"
#     model_folder_list = ["movielens_din", "movielens_gru4rec", "movielens_sasrec"]
#     # model_folder_list = ["movielens_gru4rec", "movielens_sasrec"]
#     type_folder = "base"
#     pt_folder_list = ["hist_embed.pt", "target_embed.pt"]
#     # pt_folder_list = ["target_embed.pt"]
#     for model in model_folder_list:
#         for pt in pt_folder_list:
#             print("{} {}".format(model, pt))
#             pt_path = os.path.join(chechpoint_folder, model, type_folder, pt)
#             x = torch.load(pt_path)
#             # x = np.array(x).flatten()
#             print(len(x))
#             X = np.array([np.array(i.cpu()).reshape(-1) for i in x])
#             # import ipdb
#             # ipdb.set_trace()
#             # print(X.size())
#             # X = StandardScaler().fit_transform(X)
#
#             db = DBSCAN(eps=0.1, min_samples=10).fit(X)
#             # db = DBSCAN(eps=1, min_samples=20).fit(X)
#
#             print(len(db.labels_))
#             print(db.labels_)
#             # print(db.size())
#             # print(db)
#
#             return db.labels_

if __name__ == '__main__':
    # list_a = ["a", "b", "c"]
    # print(list_a * 5)
    # print(list_a + list_a + list_a + list_a + list_a)
    a = np.array([0, 0, 1, 5, 6, 9, 0])
    print(a != 0)
    b = np.array([1, 2, 3, 4, 5, 6, 7])
    print(b[a != 0])
