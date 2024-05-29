import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random


class PCGrad():
    def __init__(self, optimizer_list, reduction='mean'):
        # self._optim, self._reduction = optimizer, reduction
        self._optim_list, self._reduction = optimizer_list, reduction
        return

    @property
    def optimizer(self):
        # return self._optim
        return self._optim_list

    def zero_grad(self, group_index):
        '''
        clear the gradient of the parameters
        '''

        # return self._optim.zero_grad(set_to_none=True)
        return self._optim_list[group_index].zero_grad(set_to_none=True)
        # return [optim.zero_grad(set_to_none=True) in self._optim_list]

    def step(self):
        '''
        update the parameters with the gradient
        '''

        # return self._optim.step()
        for group_index, optim in enumerate(self._optim_list):
            self._optim_list[group_index].step()
        return

    def pc_backward(self, objectives):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''

        # grads, shapes, has_grads = self._pack_grad(objectives)
        # pc_grad = self._project_conflicting(grads, has_grads)
        # pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        # self._set_grad(pc_grad)

        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad_list, pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad_list = self._unflatten_grad(pc_grad_list, shapes[0])
        self._set_grad(pc_grad_list)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        # print("*" * 100)
        # print(grads)
        # print(has_grads)
        # print(shared)
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        # for g_i in pc_grad:
        #     random.shuffle(grads)
        #     for g_j in grads:
        #         g_i_g_j = torch.dot(g_i, g_j)
        #         if g_i_g_j < 0:
        #             g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
        for i, g_i in enumerate(pc_grad):
            # random.shuffle(grads)
            for j, g_j in enumerate(grads):
                g_i_g_j = torch.dot(g_i, g_j)
                # print("i={}, j={}".format(i, j))
                # print(g_i_g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
                    # print("i={}, j={}".format(i, j))
                    # if i == j:
                    #     print("&&& " * 50)
                    #     print(g_i == g_j)
                    pc_grad[i] = g_i
                    # if i == j:
                    #     print(g_i == g_j)
                    #     print("&&& " * 50)

        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).sum(dim=0)
        else: exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        # print("-+-")
        # print(torch.stack(pc_grad).size())
        # print(merged_grad.size())
        return pc_grad, merged_grad

    # def _set_grad(self, grads):
    def _set_grad(self, grads_list):
        '''
        set the modified gradients to the network
        '''
        for group_index, _optim in enumerate(self._optim_list):
            idx = 0
            # for group in self._optim.param_groups:
            # for group in _optim.param_groups:
            for group in self._optim_list[group_index].param_groups:
                for p in group['params']:
                    # if p.grad is None: continue
                    grads = grads_list[group_index]
                    p.grad = grads[idx]
                    idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''
        # grads_list, shapes_list, has_grads_list = [], [], []
        # for group_index, _optim in enumerate(self._optim_list):
        #     grads, shapes, has_grads = [], [], []
        #     for obj in objectives:
        #         self._optim_list[group_index].zero_grad(set_to_none=True)
        #         # self._optim_list.zero_grad(set_to_none=True)
        #         obj.backward(retain_graph=True)
        #         grad_list, shape_list, has_grad_list = self._retrieve_grad()
        #         grads.append(self._flatten_grad(grad_list, shape_list))
        #         has_grads.append(self._flatten_grad(has_grad_list, shape_list))
        #         shapes.append(shape)
        #     grads_list.append(grads)
        #     shapes_list.append(shapes)
        #     has_grads_list.append(has_grads)
        # return grads_list, shapes_list, has_grads_list
        grads, shapes, has_grads = [], [], []
        for group_index, obj in enumerate(objectives):
            self._optim_list[group_index].zero_grad(set_to_none=True)
            # self._optim_list.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad(group_index)
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    # def _unflatten_grad(self, grads, shapes):
    def _unflatten_grad(self, grads_list, shapes):
        unflatten_grad_list = []
        for grads in grads_list:
            unflatten_grad, idx = [], 0
            for shape in shapes:
                length = np.prod(shape)
                unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
                idx += length
            unflatten_grad_list.append(unflatten_grad)
        return unflatten_grad_list
    # def _unflatten_grad(self, grads, shapes):
    #     unflatten_grad, idx = [], 0
    #     for shape in shapes:
    #         length = np.prod(shape)
    #         unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
    #         idx += length
    #     return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        # flatten_grad_list = []
        # for group_index, optim in self._optim_list:
        #     flatten_grad = torch.cat([g.flatten() for g in grads[group_index]])
        #     flatten_grad_list.append(flatten_grad)
        # return flatten_grad_list
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self, group_index):
        '''
        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''
        grad, shape, has_grad = [], [], []
        for group in self._optim_list[group_index].param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 4)

    def forward(self, x):
        return self._linear(x)


class MultiHeadTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 2)
        self._head1 = nn.Linear(2, 4)
        self._head2 = nn.Linear(2, 4)

    def forward(self, x):
        feat = self._linear(x)
        return self._head1(feat), self._head2(feat)


if __name__ == '__main__':

    # fully shared network test
    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)
    net = TestNet()
    y_pred = net(x)
    pc_adam = PCGrad(optim.Adam(net.parameters()))
    pc_adam.zero_grad()
    loss1_fn, loss2_fn = nn.L1Loss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred, y), loss2_fn(y_pred, y)

    pc_adam.pc_backward([loss1, loss2])
    for p in net.parameters():
        print(p.grad)

    print('-' * 80)
    # seperated shared network test

    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)
    net = MultiHeadTestNet()
    y_pred_1, y_pred_2 = net(x)
    pc_adam = PCGrad(optim.Adam(net.parameters()))
    pc_adam.zero_grad()
    loss1_fn, loss2_fn = nn.MSELoss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred_1, y), loss2_fn(y_pred_2, y)

    pc_adam.pc_backward([loss1, loss2])
    # for p in net.parameters():
    #     print("+" * 50)
    #     print(p)
    #     print(p.grad)

