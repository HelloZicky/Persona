import random
import numpy as np
import torch

# mini_grad_list_ = np.array([[3, 2, 1], [1, 2, 3], [3, 2, 1], [1, 2, 3], [3, 2, 1], [1, 2, 3]])
# # a = np.argmin(mini_grad_list_, 1).reshape(-1)
# # a = np.argmin(mini_grad_list_, 1).reshape(-1)
# a = np.argmin(mini_grad_list_, 1)
# # print(a.shape)
# # print(a.shape)
# # print(mini_grad_list_.shape)
# # print(mini_grad_list_[:, a].shape)
# # print(mini_grad_list_[a, :].shape)
# # print(mini_grad_list_[a].shape)
# print(a)
# print(mini_grad_list_[a])
# # print(torch.from_numpy(mini_grad_list_)[a[:], :])
# print(mini_grad_list_.argmin(1))
# # print(mini_grad_list_[:, [2, 0]])
# # print(mini_grad_list_[[0, 1, 2, 3, 4, 5], [2, 0, 2, 0, 2, 0]])
# print(mini_grad_list_[range(len(mini_grad_list_.argmin(1))), mini_grad_list_.argmin(1)])
# # print(torch.from_numpy(mini_grad_list_)[mini_grad_list_.argmin(1)])
# # print(mini_grad_list_)
# # print(mini_grad_list_[a] == mini_grad_list_)
# # print(mini_grad_list_.flatten().shape)

# low = -1
# mini_grad_list_ = []
# for step in range(20):
#     mini_grad_list = []
#     for group_index in range(5):
#         mini_grad_list.append(np.random.randint(low, high=1, size=(1024), dtype="l"))
#         # mini_grad_list.append(torch.from_numpy(np.random.randint(low, high=1, size=(1024, 1), dtype="l")))
#     # mini_grad_list.reverse()
#     # print(np.array(mini_grad_list).shape)
#     # print(np.array(mini_grad_list).transpose((1, 0, 2)).shape)
#     print(np.array(mini_grad_list).transpose((1, 0)).shape)
#     # mini_grad_list_.append(mini_grad_list)
#     # mini_grad_list_.append(np.array(mini_grad_list).transpose((1, 0, 2)))
#     mini_grad_list_.extend(np.array(mini_grad_list).transpose((1, 0)))
#
# print(np.array(mini_grad_list_).shape)
# # mini_grad_list_ = np.stack(mini_grad_list_, 1)
# # print(np.array(mini_grad_list_).shape)
# pred_list = np.random.randint(low, high=1, size=(1024 * 20, 5), dtype="l")
# print(np.array(mini_grad_list_).shape)
# print(np.argmin(mini_grad_list_, 1).shape)
# print(np.argmin(np.array(mini_grad_list_), 1).shape)
# print(pred_list.shape)
# print(pred_list[np.argmin(mini_grad_list_, 1)].shape)
# print(pred_list[range(len(np.argmin(mini_grad_list_, 1))), np.argmin(mini_grad_list_, 1)].shape)
# y_list = np.random.randint(low, high=1, size=(1024, 1), dtype="l")

low = -1
a = np.random.randint(low, high=1, size=(3, 1024, 32), dtype="l")
b = np.random.randint(low, high=1, size=(3, 272, 32), dtype="l")
# print(np.stack((a, b), 1).shape)
sample_num = a.shape[1] + b.shape[1]
print(torch.cat((torch.from_numpy(a.transpose((1, 0, 2))), torch.from_numpy(b.transpose((1, 0, 2)))), dim=0).view(sample_num, -1).shape)
