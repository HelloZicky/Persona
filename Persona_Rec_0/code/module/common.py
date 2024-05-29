"""
Common modules
"""
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from . import initializer
from . import self_attention


class StackedDense(torch.nn.Module):
    def __init__(self, in_dimension, units, activation_fns):
        super(StackedDense, self).__init__()

        modules = []
        units = [in_dimension] + list(units)
        for i in range(1, len(units)):
            linear = torch.nn.Linear(units[i-1], units[i], bias=True)
            initializer.default_weight_init(linear.weight)
            initializer.default_bias_init(linear.bias)
            modules.append(linear)

            if activation_fns[i-1] is not None:
                modules.append(activation_fns[i-1]())

        self.net = torch.nn.Sequential(*modules)

    def __setitem__(self, k, v):
        self.k = v

    def forward(self, x):
        return self.net(x)


class Linear(torch.nn.Module):
    def __init__(self, in_dimension, out_dimension, bias):
        super(Linear, self).__init__()
        self.net = torch.nn.Linear(in_dimension, out_dimension, bias)
        initializer.default_weight_init(self.net.weight)
        if bias:
            initializer.default_weight_init(self.net.bias)

    def __setitem__(self, k, v):
        self.k = v

    def forward(self, x):
        return self.net(x)


class HyperNetwork_FC(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC, self).__init__()

        self.modules = []
        units = [in_dimension] + list(units)
        self.activation_fns = activation_fns
        # self.trigger_sequence_len = trigger_sequence_len
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)
        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            # self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, input_size * output_size).cuda(), 2)))
            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            # self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, output_size).cuda(), 2)))
            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())

    def forward(self, x, z, sample_num=32, trigger_seq_length=30):
        units = self.units
        # z = z.view(sample_num, -1)

        user_state, _ = self._gru_cell(z)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        z = self._mlp_trans(user_state)
        # print(z.size())
        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            weight = torch.matmul(z, self.w1[index]) + self.b1[index]

            output_size = units[i]
            # print("=" * 50)
            # print(index)
            # print(self.units)
            # print(input_size)
            # print(output_size)
            # print(z.size())
            # print(self.w1[index].size())
            # print(self.b1[index].size())
            # print(weight.size())
            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)
            bias = torch.matmul(z, self.w2[index]) + self.b2[index]
            # print("-" * 50)
            # print(weight.size())
            # print(bias.size())
            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
            # x = self.activation_fns[index]()(x)
            x = self.modules[index](x) if index < len(self.modules) else x

        return x


class HyperNetwork_FC_gru(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC_gru, self).__init__()

        modules = []
        self.modules = []
        units = [in_dimension] + list(units)
        self.activation_fns = activation_fns
        # self.trigger_sequence_len = trigger_sequence_len
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )

        # self._hyper_gru_cell = torch.nn.GRU(
        #     model_conf.id_dimension,
        #     model_conf.id_dimension,
        #     batch_first=True
        # )

        self._mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)
        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            # self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, input_size * output_size).cuda(), 2)))
            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            # self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, output_size).cuda(), 2)))
            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                # modules.append(activation_fns[i - 1]())
                self.modules.append(activation_fns[i - 1]())

    def forward(self, x, z, sample_num=32, trigger_seq_length=30):
        units = self.units
        # z = z.view(sample_num, -1)

        # user_state, _ = self._gru_cell(z)
        # user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        # z = self._mlp_trans(user_state)

        # print(z.size())
        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            # z, _ = self._hyper_gru_cell(z)
            # print("-" * 50)
            # print("z.size() ", z.size())
            if i == 1:
                user_state, _ = self._gru_cell(z)
            else:
                user_state, _ = self._gru_cell(user_state)
            # print("user_state.size() ", user_state.size())  # (512, 10, 32)
            # print("_.size() ", _.size())
            user_state_ = user_state[range(user_state.shape[0]), trigger_seq_length, :]  # (512, 32)
            # print("user_state.size() ", user_state.size())
            z = self._mlp_trans(user_state_)  # (512, 32)
            # print("z.size() ", z.size())
            weight = torch.matmul(z, self.w1[index]) + self.b1[index]

            output_size = units[i]
            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)
            bias = torch.matmul(z, self.w2[index]) + self.b2[index]

            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
            # x = self.activation_fns[index]()(x)
            x = self.modules[index](x)

        return x


class HyperNetwork_FC_ood(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC_ood, self).__init__()

        self.modules = []
        units = [in_dimension] + list(units)
        self.activation_fns = activation_fns
        # self.trigger_sequence_len = trigger_sequence_len
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)
        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            # self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, input_size * output_size).cuda(), 2)))
            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            # self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, output_size).cuda(), 2)))
            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())

    def forward(self, x, z, sample_num=32, trigger_seq_length=30):
        units = self.units
        # z = z.view(sample_num, -1)
        user_state, _ = self._gru_cell(z)
        # print("-" * 50)
        # print(z.size())
        # print(user_state.size())
        # print(user_state.shape[0])
        # print(trigger_seq_length)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        z = self._mlp_trans(user_state)
        # print(z.size())
        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            weight = torch.matmul(z, self.w1[index]) + self.b1[index]

            output_size = units[i]
            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)
            bias = torch.matmul(z, self.w2[index]) + self.b2[index]

            # print("-" * 50)
            # print("i ", i)
            # print("units ", units)
            # print("input_size ", input_size)
            # print("output_size ", output_size)
            # print("x.size() ", x.size())
            # print("z.size() ", z.size())
            # print("self.w1[index].size() ", self.w1[index].size())
            # print("self.b1[index].size() ", self.b1[index].size())
            # print("weight.size() ", weight.size())
            #
            # print("self.w2[index].size() ", self.w2[index].size())
            # print("self.b2[index].size() ", self.b2[index].size())
            # print("bias.size() ", bias.size())
            #
            # print("x.size() ", x.size())
            # print("x.unsqueeze(1).size() ", x.unsqueeze(1).size())
            # print("torch.bmm(x.unsqueeze(1), weight).squeeze(1).size() ", torch.bmm(x.unsqueeze(1), weight).squeeze(1).size())
            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
            # x = self.activation_fns[index](x)
            x = self.modules[index](x) if index < len(self.modules) else x

        return x, user_state


class HyperNetwork_FC_ood_gru(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC_ood_gru, self).__init__()

        self.modules = []
        units = [in_dimension] + list(units)
        self.activation_fns = activation_fns
        # self.trigger_sequence_len = trigger_sequence_len
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)
        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            # self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, input_size * output_size).cuda(), 2)))
            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            # self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, output_size).cuda(), 2)))
            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())

    def forward(self, x, z, sample_num=32, trigger_seq_length=30):
        units = self.units

        # user_state, _ = self._gru_cell(z)
        #
        # user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        # z = self._mlp_trans(user_state)

        # print(z.size())
        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            if i == 1:
                user_state, _ = self._gru_cell(z)
            else:
                user_state, _ = self._gru_cell(user_state)
            user_state_ = user_state[range(user_state.shape[0]), trigger_seq_length, :]
            z = self._mlp_trans(user_state_)
            
            weight = torch.matmul(z, self.w1[index]) + self.b1[index]

            output_size = units[i]
            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)
            bias = torch.matmul(z, self.w2[index]) + self.b2[index]

            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
            # x = self.activation_fns[index](x)
            x = self.modules[index](x) if index < len(self.modules) else x

        return x, user_state_


class HyperNetwork_FC_hyper_attention(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, moe_num=5, expand=False):
        super(HyperNetwork_FC_hyper_attention, self).__init__()

        self.modules = []
        units = [in_dimension] + list(units)
        self.activation_fns = activation_fns
        # self.trigger_sequence_len = trigger_sequence_len
        self.batch = batch
        self.units = units
        self.w1, self.w2, self.b1, self.b2 = [], [], [], []
        self.w1_, self.w2_ = [], []
        self.output_size = []
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)
        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            # self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, input_size * output_size).cuda(), 2)))
            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, moe_num, input_size * output_size).cuda(), 2)))
            # self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size, moe_num).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))
            # self.w1_.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))

            # self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, output_size).cuda(), 2)))
            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, moe_num, output_size).cuda(), 2)))
            # self.b2.append(Parameter(torch.fmod(torch.randn(output_size, moe_num).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))
            # self.w2_.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())

    def forward(self, x, z, sample_num=32, trigger_seq_length=30):
        units = self.units
        # z = z.view(sample_num, -1)

        user_state, _ = self._gru_cell(z)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        z = self._mlp_trans(user_state)
        # print(z.size())
        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            # weight = torch.matmul(z, self_attention.aggregation(self.w1[index])) + self_attention.aggregation(self.b1[index])
            weight = torch.matmul(z, self_attention.aggregation(self.w1[index])) + self.b1[index]

            output_size = units[i]
            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)
            # if output_size
            # bias = torch.matmul(z, self_attention.aggregation(self.w2[index])) + self_attention.aggregation(self.b2[index])

            bias = torch.matmul(z, self_attention.aggregation(self.w2[index])) + self.b2[index] \
                if output_size != 1 else torch.matmul(z, self_attention.aggregation(self.w2[index]).unsqueeze(1)) + self.b2[index]

            # print("-" * 50)
            # print(self.w1_[index].size())
            # print(self_attention.aggregation(self.w1[index]).size())
            # print(self.w2_[index].size())
            # print(self_attention.aggregation(self.w2[index]).size())
            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
            # x = self.activation_fns[index](x)
            x = self.modules[index](x) if index < len(self.modules) else x

        return x


class HyperNetwork_FC_output_attention(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None):
        super(HyperNetwork_FC_output_attention, self).__init__()

        self.modules = []
        units = [in_dimension] + list(units)
        self.activation_fns = activation_fns
        # self.trigger_sequence_len = trigger_sequence_len
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)
        for i in range(1, len(units)):
            if i == 1:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            # self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, input_size * output_size).cuda(), 2)))
            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, moe_num, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            # self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, output_size).cuda(), 2)))
            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, moe_num, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())

    def forward(self, x, z, sample_num=32, trigger_seq_length=30):
        units = self.units
        # z = z.view(sample_num, -1)

        user_state, _ = self._gru_cell(z)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        z = self._mlp_trans(user_state)
        # print(z.size())
        for i in range(1, len(units)):
            index = i - 1
            if i == 1:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            weight = torch.matmul(z, self.w1[index]) + self.b1[index]

            output_size = units[i]
            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)
            bias = torch.matmul(z, self.w2[index]) + self.b2[index]

            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
            # x = self.activation_fns[index](x)
            x = self.modules[index](x) if index < len(self.modules) else x

        x = self_attention.aggregation(x)

        return x


class HyperNetwork_FC_apg(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False, N=64, M=32, K=16):
        super(HyperNetwork_FC_apg, self).__init__()

        self.modules = []
        units = [in_dimension] + list(units)
        self.activation_fns = activation_fns
        # self.trigger_sequence_len = trigger_sequence_len
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)
        self.expand = expand

        modules_in = []
        modules_out = []
        modules = []
        # self.net = torch.nn.Sequential(*modules)
        # print("*" * 50)
        # print(units)
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]
            # print("=" * 50)
            linear_in = torch.nn.Linear(input_size, K, bias=True)
            initializer.default_weight_init(linear_in.weight)
            initializer.default_bias_init(linear_in.bias)
            modules_in.append(linear_in)
            # print(linear_in.weight.size())

            linear_out = torch.nn.Linear(K, output_size, bias=True)
            initializer.default_weight_init(linear_out.weight)
            initializer.default_bias_init(linear_out.bias)
            modules_out.append(linear_out)
            # print(linear_out.weight.size())
            # if activation_fns[i - 1] is not None:
            #     modules.append(activation_fns[i - 1]())

            # self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, input_size * output_size).cuda(), 2)))
            # self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            # self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))
            # self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, in_dimension * K * K).cuda(), 2)))
            # self.b1.append(Parameter(torch.fmod(torch.randn(in_dimension * K * K).cuda(), 2)))
            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, K * K).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(K * K).cuda(), 2)))

            # self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension * self.trigger_sequence_len, output_size).cuda(), 2)))
            # self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            # self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))
            # self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, K * K).cuda(), 2)))
            # self.b2.append(Parameter(torch.fmod(torch.randn(K * K).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                modules.append(activation_fns[i - 1]())
            else:
                modules.append(None)

        self.K = K
        # self.modules_in = modules_in
        self.modules_in = torch.nn.Sequential(*modules_in)
        # self.modules_out = modules_out
        self.modules_out = torch.nn.Sequential(*modules_out)
        self.modules = modules

    def forward(self, x, z, sample_num=32, trigger_seq_length=30):
        units = self.units
        # z = z.view(sample_num, -1)

        user_state, _ = self._gru_cell(z)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        z = self._mlp_trans(user_state)
        # print(z.size())
        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            weight = torch.matmul(z, self.w1[index]) + self.b1[index]

            output_size = units[i]
            if not self.batch:
                # weight = weight.view(input_size, output_size)
                weight = weight.view(self.K, self.K)
            else:
                # weight = weight.view(sample_num, input_size, output_size)
                weight = weight.view(sample_num, self.K, self.K)
            # bias = torch.matmul(z, self.w2[index]) + self.b2[index]
            # print(x.device)
            # print(self.modules_in[index].device)
            # print("-" * 50)
            # print(x.size())
            # print(self.modules_in[index].weight.size())
            x = self.modules_in[index](x)
            # print(x.size())
            # x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1)
            # print(x.size())
            x = self.modules_out[index](x)
            # print(x.size())

            # if self.modules[index] is not None:
            #     x = self.modules[index](x)
            # x = self.activation_fns[index](x)
            x = self.modules[index](x) if index < len(self.modules) else x

        return x


class HyperNetwork_FC_grad(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC_grad, self).__init__()

        self.modules = []
        units = [in_dimension] + list(units)
        # modules = []
        self.activation_fns = activation_fns
        # units = [in_dimension] + list(units)
        units = list(units)
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)
        self.expand = expand
        # self.z_finetune = []
        self.z_finetune = nn.ModuleList()
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            self.z_finetune.append(
                nn.Linear(model_conf.id_dimension, model_conf.id_dimension)
            )

            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())

    # def forward(self, x, z_list, sample_num=32, pretrain_model=None, grad_norm=1):
    # def forward(self, x, z, sample_num=1024, trigger_seq_length=30, pretrain_model=None, grad_norm=1, return_grad=False):
    def forward(self, x, z, sample_num=1024, trigger_seq_length=30, pretrain_model=None, grad_norm=1, return_grad=False, center_z=None, grad_norm_=0.5, dynamic_partition=False):
        # print("indicator")
        # print("----center=base generate=grad----")
        units = self.units
        # z = z.view(sample_num, -1)

        user_state, _ = self._gru_cell(z)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        z_ = self._mlp_trans(user_state)
        # print(z.size())
        grad_list = []
        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            # z = z_list[index]
            # import ipdb
            # ipdb.set_trace()
            z = self.z_finetune[index](z_)
            weight = torch.matmul(z, self.w1[index]) + self.b1[index]
            # grad_list.append(torch.clip(weight, min=-grad_norm, max=grad_norm))
            grad_list.append(z)

            output_size = units[i]
            # print("=" * 50)
            # print(index)
            # print(self.units)
            # print(input_size)
            # print(output_size)
            # print(z.size())
            # print(self.w1[index].size())
            # print(self.b1[index].size())
            # print(weight.size())
            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)

            bias = torch.matmul(z, self.w2[index]) + self.b2[index]
            if pretrain_model is not None:
                # print("=" * 50)
                # print(torch.clip(weight, min=-0.01, max=0.01).size())
                # print(pretrain_model["linear{}.weight".format(index + 3)].size())
                # print(pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1).size())
                # print(pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1)[:sample_num].size())
                # print("*" * 50)
                # print(torch.clip(weight, min=-grad_norm, max=grad_norm).size())
                # print(pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(1024, 1, 1).permute(0, 2, 1).size())
                # print("-" * 50)
                # print(torch.clip(bias, min=-grad_norm, max=grad_norm).size())
                # print(pretrain_model["_classifier.net.{}.bias".format(index * 2)].size())
                # print(pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(1024, 1, 1).permute(0, 2, 1).size())
                # weight = torch.clip(weight, min=-grad_norm, max=grad_norm) + pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1)[:sample_num]
                # bias = torch.clip(bias, min=-grad_norm, max=grad_norm) + pretrain_model["linear{}.bias".format(index + 3)].squeeze(1).squeeze(1)[:sample_num]
                # print(torch.clip(weight, min=-grad_norm, max=grad_norm).size())
                # print(weight.size())
                # print(torch.clip(bias, min=-grad_norm, max=grad_norm).size())
                # print(bias.size())
                # print(pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(1024, 1, 1).permute(0, 2, 1).size())
                weight = torch.clip(weight, min=-grad_norm, max=grad_norm) + pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1)
                bias = torch.clip(bias, min=-grad_norm, max=grad_norm) + pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)

            # x = self.activation_fns[index](torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias) if self.activation_fns[index] is not None else (torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias)
            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
            x = self.modules[index](x) if index < len(self.modules) else x

        if return_grad:
            return x, grad_list
        else:
            return x


class HyperNetwork_FC_grad_gru(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC_grad_gru, self).__init__()

        self.modules = []
        units = [in_dimension] + list(units)
        # modules = []
        self.activation_fns = activation_fns
        # units = [in_dimension] + list(units)
        units = list(units)
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self._mlp_trans_init = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )

        self._mlp_trans = nn.ModuleList([StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )] * len(units))

        self._mlp_trans2 = nn.ModuleList([StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )] * len(units))

        self._gru_cell_seq = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        initializer.default_weight_init(self._gru_cell_seq.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell_seq.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell_seq.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell_seq.bias_hh_l0)

        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            len(units) - 1,
            # batch_first=True
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)

        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())

        # def forward(self, x, z_list, sample_num=32, pretrain_model=None, grad_norm=1):

    def forward(self, x, z, sample_num=1024, trigger_seq_length=30, pretrain_model=None, grad_norm=1,
                return_grad=False):
        # print("indicator")
        # print("----center=base generate=grad_gru----")
        units = self.units
        # z = z.view(sample_num, -1)

        # user_state, _ = self._gru_cell(z)
        user_state, _ = self._gru_cell_seq(z)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        # z = self._mlp_trans(user_state)
        z = self._mlp_trans_init(user_state)
        # print(z.size())
        grad_list = []

        z_list = []
        for i in range(1, len(units)):
            # index = i - 1
            # z = self._mlp_trans[index](z)
            z_list.append(z)

        z_ = torch.stack(z_list)
        # _, z_gru = self._gru_cell(z.unsqueeze(0), z_)
        _, z_gru = self._gru_cell(z_)

        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            # z = z_list[index]
            # import ipdb
            # ipdb.set_trace()
            z = z_gru[index].squeeze(0)
            z = nn.Tanh()(self._mlp_trans2[index](z))
            grad_list.append(z)

            weight = torch.matmul(z, self.w1[index]) + self.b1[index]
            # grad_list.append(torch.clip(weight, min=-grad_norm, max=grad_norm))
            # grad_list.append(weight)

            output_size = units[i]
            # print("=" * 50)
            # print(index)
            # print(self.units)
            # print(input_size)
            # print(output_size)
            # print(z.size())
            # print(self.w1[index].size())
            # print(self.b1[index].size())
            # print(weight.size())
            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)

            bias = torch.matmul(z, self.w2[index]) + self.b2[index]
            if pretrain_model is not None:
                # print("=" * 50)
                # print(torch.clip(weight, min=-0.01, max=0.01).size())
                # print(pretrain_model["linear{}.weight".format(index + 3)].size())
                # print(pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1).size())
                # print(pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1)[:sample_num].size())
                # print("*" * 50)
                # print(torch.clip(weight, min=-grad_norm, max=grad_norm).size())
                # print(pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(1024, 1, 1).permute(0, 2, 1).size())
                # print("-" * 50)
                # print(torch.clip(bias, min=-grad_norm, max=grad_norm).size())
                # print(pretrain_model["_classifier.net.{}.bias".format(index * 2)].size())
                # print(pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(1024, 1, 1).permute(0, 2, 1).size())
                # weight = torch.clip(weight, min=-grad_norm, max=grad_norm) + pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1)[:sample_num]
                # bias = torch.clip(bias, min=-grad_norm, max=grad_norm) + pretrain_model["linear{}.bias".format(index + 3)].squeeze(1).squeeze(1)[:sample_num]
                # print(torch.clip(weight, min=-grad_norm, max=grad_norm).size())
                # print(weight.size())
                # print(torch.clip(bias, min=-grad_norm, max=grad_norm).size())
                # print(bias.size())
                # print(pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(1024, 1, 1).permute(0, 2, 1).size())

                weight = torch.clip(weight, min=-grad_norm, max=grad_norm) + pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1)
                bias = torch.clip(bias, min=-grad_norm, max=grad_norm) + pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)
                # weight = weight + pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1,
                #                                                                                        1).permute(0, 2,
                #                                                                                                   1)
                # bias = bias + pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)

            # x = self.activation_fns[index](torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias) if self.activation_fns[index] is not None else (torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias)
            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
            x = self.modules[index](x) if index < len(self.modules) else x

        if return_grad:
            return x, grad_list
        else:
            return x

class HyperNetwork_FC_grad_gru_c(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC_grad_gru_c, self).__init__()

        self.modules = []
        units = [in_dimension] + list(units)
        # modules = []
        self.activation_fns = activation_fns
        # units = [in_dimension] + list(units)
        units = list(units)
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self._mlp_trans_init = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )

        self._mlp_trans = nn.ModuleList([StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )] * len(units))

        self._mlp_trans2 = nn.ModuleList([StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )] * len(units))

        self._gru_cell_seq = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        initializer.default_weight_init(self._gru_cell_seq.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell_seq.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell_seq.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell_seq.bias_hh_l0)

        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            len(units) - 1,
            # batch_first=True
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)

        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())

        # def forward(self, x, z_list, sample_num=32, pretrain_model=None, grad_norm=1):

    def forward(self, x, z, sample_num=1024, trigger_seq_length=30, pretrain_model=None, grad_norm=1,
                # return_grad=False, center_z=None):
                return_grad=False, center_z=None):
        # print("----center=base generate=grad_gru----")
        units = self.units
        # z = z.view(sample_num, -1)

        # user_state, _ = self._gru_cell(z)
        user_state, _ = self._gru_cell_seq(z)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        # z = self._mlp_trans(user_state)
        z = self._mlp_trans_init(user_state)
        # print(z.size())
        grad_list = []

        z_list = []
        for i in range(1, len(units)):
            # index = i - 1
            # z = self._mlp_trans[index](z)
            z_list.append(z)

        z_ = torch.stack(z_list)
        # _, z_gru = self._gru_cell(z.unsqueeze(0), z_)
        _, z_gru = self._gru_cell(z_)

        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            # z = z_list[index]
            # import ipdb
            # ipdb.set_trace()
            z = z_gru[index].squeeze(0)
            z = nn.Tanh()(self._mlp_trans2[index](z))
            grad_list.append(z)

            weight = torch.matmul(z, self.w1[index]) + self.b1[index]
            # grad_list.append(torch.clip(weight, min=-grad_norm, max=grad_norm))
            # grad_list.append(weight)

            output_size = units[i]
            # print("=" * 50)
            # print(index)
            # print(self.units)
            # print(input_size)
            # print(output_size)
            # print(z.size())
            # print(self.w1[index].size())
            # print(self.b1[index].size())
            # print(weight.size())
            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)

            bias = torch.matmul(z, self.w2[index]) + self.b2[index]
            if pretrain_model is not None:
                # print("=" * 50)
                # print(torch.clip(weight, min=-0.01, max=0.01).size())
                # print(pretrain_model["linear{}.weight".format(index + 3)].size())
                # print(pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1).size())
                # print(pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1)[:sample_num].size())
                # print("*" * 50)
                # print(torch.clip(weight, min=-grad_norm, max=grad_norm).size())
                # print(pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(1024, 1, 1).permute(0, 2, 1).size())
                # print("-" * 50)
                # print(torch.clip(bias, min=-grad_norm, max=grad_norm).size())
                # print(pretrain_model["_classifier.net.{}.bias".format(index * 2)].size())
                # print(pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(1024, 1, 1).permute(0, 2, 1).size())
                # weight = torch.clip(weight, min=-grad_norm, max=grad_norm) + pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1)[:sample_num]
                # bias = torch.clip(bias, min=-grad_norm, max=grad_norm) + pretrain_model["linear{}.bias".format(index + 3)].squeeze(1).squeeze(1)[:sample_num]
                # print(torch.clip(weight, min=-grad_norm, max=grad_norm).size())
                # print(weight.size())
                # print(torch.clip(bias, min=-grad_norm, max=grad_norm).size())
                # print(bias.size())
                # print(pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(1024, 1, 1).permute(0, 2, 1).size())

                weight = torch.clip(weight, min=-grad_norm, max=grad_norm) + pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1)
                bias = torch.clip(bias, min=-grad_norm, max=grad_norm) + pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)
                # weight = weight + pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1,
                #                                                                                        1).permute(0, 2,
                #                                                                                                   1)
                # bias = bias + pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)

            # x = self.activation_fns[index](torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias) if self.activation_fns[index] is not None else (torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias)
            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
            x = self.modules[index](x) if index < len(self.modules) else x

        if return_grad:
            return x, grad_list
        else:
            return x


class HyperNetwork_FC_grad_gru_center(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC_grad_gru_center, self).__init__()

        self.modules = []
        units = [in_dimension] + list(units)
        # modules = []
        self.activation_fns = activation_fns
        # units = [in_dimension] + list(units)
        units = list(units)
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self._mlp_trans_init = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )

        self._mlp_trans = nn.ModuleList([StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )] * len(units))

        self._mlp_trans2 = nn.ModuleList([StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )] * len(units))

        self._gru_cell_seq = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        initializer.default_weight_init(self._gru_cell_seq.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell_seq.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell_seq.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell_seq.bias_hh_l0)

        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            len(units) - 1,
            # batch_first=True
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)

        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())

        # def forward(self, x, z_list, sample_num=32, pretrain_model=None, grad_norm=1):

    def forward(self, x, z, sample_num=1024, trigger_seq_length=30, pretrain_model=None, grad_norm=1,
                return_grad=False, center_z=None, grad_norm_=0.5, dynamic_partition=False):
        # grad_norm_ = 0.5
        center_z = center_z.repeat(sample_num, 1)
        units = self.units
        # z = z.view(sample_num, -1)

        # user_state, _ = self._gru_cell(z)
        user_state, _ = self._gru_cell_seq(z)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        # z = self._mlp_trans(user_state)
        z = self._mlp_trans_init(user_state)
        # print(z.size())
        grad_list = []

        z_list = []
        for i in range(1, len(units)):
            # index = i - 1
            # z = self._mlp_trans[index](z)
            z_list.append(z)

        z_ = torch.stack(z_list)
        # _, z_gru = self._gru_cell(z.unsqueeze(0), z_)
        _, z_gru = self._gru_cell(z_)

        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            # z = z_list[index]
            # import ipdb
            # ipdb.set_trace()
            z = z_gru[index].squeeze(0)
            z = nn.Tanh()(self._mlp_trans2[index](z))
            grad_list.append(z)

            # weight = torch.matmul(z, self.w1[index]) + self.b1[index]
            weight = torch.clip(torch.matmul(z, self.w1[index]) + self.b1[index], min=-grad_norm_, max=grad_norm_)
            weight_center = torch.matmul(center_z, self.w1[index]) + self.b1[index]
            weight = weight + weight_center
            # grad_list.append(torch.clip(weight, min=-grad_norm, max=grad_norm))
            # grad_list.append(weight)

            output_size = units[i]
            # print("=" * 50)
            # print(index)
            # print(self.units)
            # print(input_size)
            # print(output_size)
            # print(z.size())
            # print(self.w1[index].size())
            # print(self.b1[index].size())
            # print(weight.size())
            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)

            # bias = torch.matmul(z, self.w2[index]) + self.b2[index]
            bias = torch.clip(torch.matmul(z, self.w2[index]) + self.b2[index], min=-grad_norm_, max=grad_norm_)
            bias_center = torch.matmul(center_z, self.w2[index]) + self.b2[index]
            bias = bias + bias_center
            if pretrain_model is not None:
                # print("=" * 50)
                # print(torch.clip(weight, min=-0.01, max=0.01).size())
                # print(pretrain_model["linear{}.weight".format(index + 3)].size())
                # print(pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1).size())
                # print(pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1)[:sample_num].size())
                # print("*" * 50)
                # print(torch.clip(weight, min=-grad_norm, max=grad_norm).size())
                # print(pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(1024, 1, 1).permute(0, 2, 1).size())
                # print("-" * 50)
                # print(torch.clip(bias, min=-grad_norm, max=grad_norm).size())
                # print(pretrain_model["_classifier.net.{}.bias".format(index * 2)].size())
                # print(pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(1024, 1, 1).permute(0, 2, 1).size())
                # weight = torch.clip(weight, min=-grad_norm, max=grad_norm) + pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1)[:sample_num]
                # bias = torch.clip(bias, min=-grad_norm, max=grad_norm) + pretrain_model["linear{}.bias".format(index + 3)].squeeze(1).squeeze(1)[:sample_num]
                # print(torch.clip(weight, min=-grad_norm, max=grad_norm).size())
                # print(weight.size())
                # print(torch.clip(bias, min=-grad_norm, max=grad_norm).size())
                # print(bias.size())
                # print(pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(1024, 1, 1).permute(0, 2, 1).size())

                weight = torch.clip(weight, min=-grad_norm, max=grad_norm) + pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1)
                bias = torch.clip(bias, min=-grad_norm, max=grad_norm) + pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)
                # weight = weight + pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1,
                #                                                                                        1).permute(0, 2,
                #                                                                                                   1)
                # bias = bias + pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)

            # x = self.activation_fns[index](torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias) if self.activation_fns[index] is not None else (torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias)
            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
            x = self.modules[index](x) if index < len(self.modules) else x

        if return_grad:
            return x, grad_list
        else:
            return x


class HyperNetwork_FC_grad_gru_center_dynamic(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC_grad_gru_center_dynamic, self).__init__()

        self.modules = []
        units = [in_dimension] + list(units)
        # modules = []
        self.activation_fns = activation_fns
        # units = [in_dimension] + list(units)
        units = list(units)
        self.batch = batch
        self.units = units
        self.w1_group = []
        self.w2_group = []
        self.b1_group = []
        self.b2_group = []

        self.w1_sample = []
        self.w2_sample = []
        self.b1_sample = []
        self.b2_sample = []
        self.output_size = []
        self.id_dimension = model_conf.id_dimension
        self._mlp_trans_init = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )

        self._mlp_trans = nn.ModuleList([StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )] * len(units))

        self._mlp_trans2 = nn.ModuleList([StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )] * len(units))

        self._gru_cell_seq = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        initializer.default_weight_init(self._gru_cell_seq.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell_seq.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell_seq.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell_seq.bias_hh_l0)

        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            len(units) - 1,
            # batch_first=True
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)

        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            self.w1_group.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1_group.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            self.w2_group.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2_group.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            self.w1_sample.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1_sample.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            self.w2_sample.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2_sample.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())

        # def forward(self, x, z_list, sample_num=32, pretrain_model=None, grad_norm=1):

    def forward(self, x, z, sample_num=1024, trigger_seq_length=30, pretrain_model=None, grad_norm=1,
                return_grad=False, center_z=None, grad_norm_=0.5, dynamic_partition=False, stage=0
                ):
        # if return_mini_grad:
        # if return_grad:
        if dynamic_partition:
            # print("indicator")
            # print("----dynamic_partition----")
            # grad_norm_ = 0.5
            # center_z = center_z.unsqueeze(0).repeat(sample_num, 1, 1)
            center_z = center_z.repeat(sample_num, 1)
            # print("-" * 50)
            # print("center_z.size()", center_z.size())
            units = self.units
            # z = z.view(sample_num, -1)

            # user_state, _ = self._gru_cell(z)
            user_state, _ = self._gru_cell_seq(z)
            user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
            # z = self._mlp_trans(user_state)
            z = self._mlp_trans_init(user_state)
            # print(z.size())
            grad_list = []
            mini_grad_list = []
            mini_grad_item_list = []

            z_list = []
            for i in range(1, len(units)):
                # index = i - 1
                # z = self._mlp_trans[index](z)
                z_list.append(z)

            z_ = torch.stack(z_list)
            # _, z_gru = self._gru_cell(z.unsqueeze(0), z_)
            _, z_gru = self._gru_cell(z_)

            for i in range(1, len(units)):
                index = i - 1
                if i == 1 and self.expand:
                    input_size = units[i - 1] * 2
                else:
                    input_size = units[i - 1]
                # z = z_list[index]
                # import ipdb
                # ipdb.set_trace()
                z = z_gru[index].squeeze(0)
                z = nn.Tanh()(self._mlp_trans2[index](z))
                # grad_list.append(z)
                # grad_list.append(z.detach().cpu().numpy())
                grad_list.append(z.detach().cpu())

                id_dimension = 32
                # print(center_z[:, :, :][0])
                # print(center_z[:, :, index * self.id_dimension: (index + 1) * self.id_dimension][0])
                # weight = torch.matmul(z, self.w1_group[index]) + self.b1_group[index]
                weight_mini = torch.clip(torch.matmul(z, self.w1_group[index]) + self.b1_group[index], min=-grad_norm_, max=grad_norm_)
                # weight_center = torch.matmul(center_z[:, :, index * self.id_dimension: (index + 1) * self.id_dimension],
                #                              self.w1_group[index]) + self.b1_group[index]
                # print(center_z.size())
                # print(center_z[:, index * id_dimension: (index + 1) * id_dimension].size())
                weight_center = torch.matmul(center_z[:, index * id_dimension: (index + 1) * id_dimension],
                                             self.w1_group[index]) + self.b1_group[index]
                weight = weight_mini + weight_center
                # grad_list.append(torch.clip(weight, min=-grad_norm, max=grad_norm))
                # grad_list.append(weight)
                # mini_grad_list.append(weight_mini)
                # mini_grad_list.append(torch.norm(weight_mini, p=2))
                # print("weight_mini.size() ", weight_mini.size())
                # mini_grad_item_list.append(torch.norm(weight_mini, p=2).detach().cpu().numpy())
                # mini_grad_item_list.append(weight_mini.detach().cpu().numpy())
                # mini_grad_item_list.append(weight_mini.detach().cpu())
                # mini_grad_item_list.extend(weight_mini.detach().cpu())
                mini_grad_item_list.append(weight_mini.detach().cpu())

                output_size = units[i]
                # print("=" * 50)
                # print(index)
                # print(self.units)
                # print(input_size)
                # print(output_size)
                # print(z.size())
                # print(self.w1_group[index].size())
                # print(self.b1_group[index].size())
                # print(weight.size())
                if not self.batch:
                    weight = weight.view(input_size, output_size)
                else:
                    weight = weight.view(sample_num, input_size, output_size)

                # bias = torch.matmul(z, self.w2_group[index]) + self.b2_group[index]
                bias = torch.clip(torch.matmul(z, self.w2_group[index]) + self.b2_group[index], min=-grad_norm_, max=grad_norm_)
                bias_center = torch.matmul(center_z[:, index * id_dimension: (index + 1) * id_dimension], self.w2_group[index]) + self.b2_group[index]
                bias = bias + bias_center
                if pretrain_model is not None:
                    # print("=" * 50)
                    # print(torch.clip(weight, min=-0.01, max=0.01).size())
                    # print(pretrain_model["linear{}.weight".format(index + 3)].size())
                    # print(pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1).size())
                    # print(pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1)[:sample_num].size())
                    # print("*" * 50)
                    # print(torch.clip(weight, min=-grad_norm, max=grad_norm).size())
                    # print(pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(1024, 1, 1).permute(0, 2, 1).size())
                    # print("-" * 50)
                    # print(torch.clip(bias, min=-grad_norm, max=grad_norm).size())
                    # print(pretrain_model["_classifier.net.{}.bias".format(index * 2)].size())
                    # print(pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(1024, 1, 1).permute(0, 2, 1).size())
                    # weight = torch.clip(weight, min=-grad_norm, max=grad_norm) + pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1)[:sample_num]
                    # bias = torch.clip(bias, min=-grad_norm, max=grad_norm) + pretrain_model["linear{}.bias".format(index + 3)].squeeze(1).squeeze(1)[:sample_num]
                    # print(torch.clip(weight, min=-grad_norm, max=grad_norm).size())
                    # print(weight.size())
                    # print(torch.clip(bias, min=-grad_norm, max=grad_norm).size())
                    # print(bias.size())
                    # print(pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(1024, 1, 1).permute(0, 2, 1).size())

                    weight = torch.clip(weight, min=-grad_norm, max=grad_norm) + pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1)
                    bias = torch.clip(bias, min=-grad_norm, max=grad_norm) + pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)
                    # weight = weight + pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1,
                    #                                                                                        1).permute(0, 2,
                    #                                                                                                   1)
                    # bias = bias + pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)

                # x = self.activation_fns[index](torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias) if self.activation_fns[index] is not None else (torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias)
                x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
                x = self.modules[index](x) if index < len(self.modules) else x
            # print(mini_grad_item_list[0].size())
            # print(mini_grad_item_list[1].size())
            # print(mini_grad_item_list[2].size())
            # # print(torch.Tensor(mini_grad_item_list).size())
            # # print(torch.stack(mini_grad_item_list).size())
            # # import ipdb
            # # ipdb.set_trace()
            # # print("torch.stack(mini_grad_item_list).size() ", torch.cat(mini_grad_item_list, dim=0).size())
            # print("torch.stack(mini_grad_item_list).size() ", torch.cat(mini_grad_item_list, dim=1).size())
            # print("torch.stack(mini_grad_item_list).size() ", torch.stack(mini_grad_item_list, dim=1).size())
            # print("torch.norm(torch.stack(mini_grad_item_list), p=2).size() ", torch.norm(torch.stack(mini_grad_item_list), p=2).size())
            # print("torch.stack(mini_grad_item_list).size() ", torch.cat(torch.Tensor(mini_grad_item_list), dim=1).size())
            # print("torch.stack(mini_grad_item_list).size() ", torch.stack(torch.Tensor(mini_grad_item_list), dim=0).size())
            # print("torch.norm(torch.stack(mini_grad_item_list), p=2).size() ", torch.norm(torch.stack(torch.Tensor(mini_grad_item_list)), p=2).size())
            # mini_grad_list.append(torch.norm(torch.cat(mini_grad_item_list, dim=1), p=2).detach().cpu().numpy())
            # print("=== ===")
            # print(torch.cat(mini_grad_item_list, dim=1).size())
            # print(torch.norm(torch.cat(mini_grad_item_list, dim=1), p=2).size())
            # print(torch.norm(torch.cat(mini_grad_item_list, dim=1), p=2).numpy().shape)
            # print(torch.norm(torch.cat(mini_grad_item_list, dim=1), dim=1, p=2).numpy().shape)
            # mini_grad_list = torch.norm(torch.cat(mini_grad_item_list, dim=1), p=2).detach().cpu().numpy()
            # mini_grad_list = torch.norm(torch.cat(mini_grad_item_list, dim=1), dim=1, p=2).detach().cpu().numpy().reshape(sample_num, -1)
            # 1024
            mini_grad_list = torch.norm(torch.cat(mini_grad_item_list, dim=1), dim=1, p=2).detach().cpu().numpy().reshape(sample_num)
            # mini_grad_list = torch.norm(torch.cat(mini_grad_item_list, dim=1), dim=1, p=2).detach().cpu().view(sample_num)
            # print(mini_grad_list.shape)
            # grad_list = torch.stack(grad_list).transpose(0, 1)
            # grad_list = torch.stack(grad_list).permute(1, 0, 2)
            # 1024 96
            grad_list = torch.stack(grad_list).transpose(0, 1).reshape(sample_num, -1).numpy()
            # grad_list = torch.stack(grad_list).permute(1, 0, 2).view(sample_num, -1)
            # print("--common--")
            # print(x.size())
            # # print(grad_list.size())
            # # print(mini_grad_list.size())
            # print(grad_list.shape)
            # print(mini_grad_list.shape)
            return x, grad_list, mini_grad_list

        else:
            if center_z is not None:
                if stage == 1:
                    # print("indicator")
                    # print("----partition=fix center=group generate=grad_gru----")
                    # grad_norm_ = 0.5
                    center_z = center_z.repeat(sample_num, 1)
                    units = self.units
                    # z = z.view(sample_num, -1)

                    # user_state, _ = self._gru_cell(z)
                    user_state, _ = self._gru_cell_seq(z)
                    user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
                    # z = self._mlp_trans(user_state)
                    z = self._mlp_trans_init(user_state)
                    # print(z.size())
                    grad_list = []

                    z_list = []
                    for i in range(1, len(units)):
                        # index = i - 1
                        # z = self._mlp_trans[index](z)
                        z_list.append(z)

                    z_ = torch.stack(z_list)
                    # _, z_gru = self._gru_cell(z.unsqueeze(0), z_)
                    _, z_gru = self._gru_cell(z_)

                    for i in range(1, len(units)):
                        index = i - 1
                        if i == 1 and self.expand:
                            input_size = units[i - 1] * 2
                        else:
                            input_size = units[i - 1]
                        # z = z_list[index]
                        # import ipdb
                        # ipdb.set_trace()
                        z = z_gru[index].squeeze(0)
                        z = nn.Tanh()(self._mlp_trans2[index](z))
                        grad_list.append(z)

                        id_dimension = 32
                        # weight = torch.matmul(z, self.w1_group[index]) + self.b1_group[index]
                        # grad_sample = torch.clip(torch.matmul(z, self.w1_group[index]) + self.b1_group[index], min=-grad_norm_,
                        #                     max=grad_norm_)
                        grad_sample = torch.matmul(z, self.w1_sample[index]) + self.b1_sample[index]
                        # weight_center = torch.matmul(center_z, self.w1_group[index]) + self.b1_group[index]
                        grad_center = torch.matmul(center_z[:, index * id_dimension: (index + 1) * id_dimension],
                                                     self.w1_group[index]) + self.b1_group[index]
                        grad_sample = torch.clip(grad_sample, min=-grad_norm_, max=grad_norm_)
                        grad_center = torch.clip(grad_center, min=-grad_norm, max=grad_norm)
                        # weight = grad_sample + grad_center
                        weight = grad_center
                        # grad_list.append(torch.clip(weight, min=-grad_norm, max=grad_norm))
                        # grad_list.append(weight)

                        output_size = units[i]

                        if not self.batch:
                            weight = weight.view(input_size, output_size)
                            # grad_center = grad_center.view(input_size, output_size)
                        else:
                            weight = weight.view(sample_num, input_size, output_size)
                            # grad_center = grad_center.view(sample_num, input_size, output_size)

                        # bias = torch.matmul(z, self.w2_group[index]) + self.b2_group[index]
                        # bias = torch.clip(torch.matmul(z, self.w2_group[index]) + self.b2_group[index], min=-grad_norm_, max=grad_norm_)
                        # bias_sample = torch.clip(torch.matmul(z, self.w2_sample[index]) + self.b2_sample[index], min=-grad_norm_, max=grad_norm_)
                        bias_sample = torch.matmul(z,
                                                   self.w2_sample[index]) + self.b2_sample[index]
                        bias_center = torch.matmul(center_z[:, index * id_dimension: (index + 1) * id_dimension],
                                                   self.w2_group[index]) + self.b2_group[index]
                        # bias = bias_sample + bias_center
                        bias = bias_center
                        if pretrain_model is not None:

                            # weight = torch.clip(weight, min=-grad_norm, max=grad_norm) + pretrain_model[
                            #     "_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1)
                            # bias = torch.clip(bias, min=-grad_norm, max=grad_norm) + pretrain_model[
                            #     "_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)
                            weight = weight + pretrain_model[
                                "_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1)
                            bias = bias + pretrain_model[
                                "_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)

                        x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
                        x = self.modules[index](x) if index < len(self.modules) else x

                    if return_grad:
                        return x, grad_list
                    else:
                        return x

                elif stage == 2:
                    # print("indicator")
                    # print("----partition=fix center=group generate=grad_gru----")
                    # grad_norm_ = 0.5
                    center_z = center_z.repeat(sample_num, 1)
                    units = self.units
                    # z = z.view(sample_num, -1)

                    # user_state, _ = self._gru_cell(z)
                    user_state, _ = self._gru_cell_seq(z)
                    user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
                    # z = self._mlp_trans(user_state)
                    z = self._mlp_trans_init(user_state)
                    # print(z.size())
                    grad_list = []

                    z_list = []
                    for i in range(1, len(units)):
                        # index = i - 1
                        # z = self._mlp_trans[index](z)
                        z_list.append(z)

                    z_ = torch.stack(z_list)
                    # _, z_gru = self._gru_cell(z.unsqueeze(0), z_)
                    _, z_gru = self._gru_cell(z_)

                    for i in range(1, len(units)):
                        index = i - 1
                        self.w1_group[index].requires_grad = False
                        self.b1_group[index].requires_grad = False
                        if i == 1 and self.expand:
                            input_size = units[i - 1] * 2
                        else:
                            input_size = units[i - 1]
                        # z = z_list[index]
                        # import ipdb
                        # ipdb.set_trace()
                        z = z_gru[index].squeeze(0)
                        z = nn.Tanh()(self._mlp_trans2[index](z))
                        grad_list.append(z)

                        id_dimension = 32
                        # weight = torch.matmul(z, self.w1_group[index]) + self.b1_group[index]
                        # weight_sample = torch.clip(torch.matmul(z, self.w1_group[index]) + self.b1_group[index],
                        #                     min=-grad_norm_,
                        #                     max=grad_norm_)
                        # # weight_center = torch.matmul(center_z, self.w1_group[index]) + self.b1_group[index]
                        # weight_center = torch.matmul(center_z[:, index * id_dimension: (index + 1) * id_dimension],
                        #                              self.w1_group[index]) + self.b1_group[index]
                        grad_sample = torch.matmul(z, self.w1_sample[index]) + self.b1_sample[index]
                        # weight_center = torch.matmul(center_z, self.w1_group[index]) + self.b1_group[index]
                        grad_center = torch.matmul(center_z[:, index * id_dimension: (index + 1) * id_dimension],
                                                   self.w1_group[index]) + self.b1_group[index]
                        grad_sample = torch.clip(grad_sample, min=-grad_norm_, max=grad_norm_)
                        grad_center = torch.clip(grad_center, min=-grad_norm, max=grad_norm)
                        # weight = weight_sample + weight_center
                        weight = grad_sample + grad_center
                        # grad_list.append(torch.clip(weight, min=-grad_norm, max=grad_norm))
                        # grad_list.append(weight)

                        output_size = units[i]

                        if not self.batch:
                            weight = weight.view(input_size, output_size)
                        else:
                            weight = weight.view(sample_num, input_size, output_size)

                        # bias = torch.matmul(z, self.w2_group[index]) + self.b2_group[index]
                        # bias = torch.clip(torch.matmul(z, self.w2_group[index]) + self.b2_group[index], min=-grad_norm_,
                        #                   max=grad_norm_)
                        bias_sample = torch.matmul(z, self.w2_sample[index]) + self.b2_sample[index]
                        # bias_center = torch.matmul(center_z, self.w2_group[index]) + self.b2_group[index]
                        bias_center = torch.matmul(center_z[:, index * id_dimension: (index + 1) * id_dimension],
                                                   self.w2_group[index]) + self.b2_group[index]
                        bias = bias_sample + bias_center
                        if pretrain_model is not None:
                            # weight = torch.clip(weight, min=-grad_norm, max=grad_norm) + pretrain_model[
                            weight = weight + pretrain_model[
                                "_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1)
                            # bias = torch.clip(bias, min=-grad_norm, max=grad_norm) + pretrain_model[
                            bias = bias + pretrain_model[
                                "_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)

                        x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
                        x = self.modules[index](x) if index < len(self.modules) else x

                    if return_grad:
                        return x, grad_list
                    else:
                        return x

            else:
                # print("indicator")
                # print("----partition=fix center=base generate=grad_gru----")
                units = self.units
                # z = z.view(sample_num, -1)

                # user_state, _ = self._gru_cell(z)
                user_state, _ = self._gru_cell_seq(z)
                user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
                # z = self._mlp_trans(user_state)
                z = self._mlp_trans_init(user_state)
                # print(z.size())
                grad_list = []

                z_list = []
                for i in range(1, len(units)):
                    # index = i - 1
                    # z = self._mlp_trans[index](z)
                    z_list.append(z)

                z_ = torch.stack(z_list)
                # _, z_gru = self._gru_cell(z.unsqueeze(0), z_)
                _, z_gru = self._gru_cell(z_)

                for i in range(1, len(units)):
                    index = i - 1
                    if i == 1 and self.expand:
                        input_size = units[i - 1] * 2
                    else:
                        input_size = units[i - 1]
                    # z = z_list[index]
                    # import ipdb
                    # ipdb.set_trace()
                    z = z_gru[index].squeeze(0)
                    z = nn.Tanh()(self._mlp_trans2[index](z))
                    grad_list.append(z)

                    weight = torch.matmul(z, self.w1_group[index]) + self.b1_group[index]
                    # grad_list.append(torch.clip(weight, min=-grad_norm, max=grad_norm))
                    # grad_list.append(weight)

                    output_size = units[i]
                    # print("=" * 50)
                    # print(index)
                    # print(self.units)
                    # print(input_size)
                    # print(output_size)
                    # print(z.size())
                    # print(self.w1_group[index].size())
                    # print(self.b1_group[index].size())
                    # print(weight.size())
                    if not self.batch:
                        weight = weight.view(input_size, output_size)
                    else:
                        weight = weight.view(sample_num, input_size, output_size)

                    bias = torch.matmul(z, self.w2_group[index]) + self.b2_group[index]
                    if pretrain_model is not None:
                        # print("=" * 50)
                        # print(torch.clip(weight, min=-0.01, max=0.01).size())
                        # print(pretrain_model["linear{}.weight".format(index + 3)].size())
                        # print(pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1).size())
                        # print(pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1)[:sample_num].size())
                        # print("*" * 50)
                        # print(torch.clip(weight, min=-grad_norm, max=grad_norm).size())
                        # print(pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(1024, 1, 1).permute(0, 2, 1).size())
                        # print("-" * 50)
                        # print(torch.clip(bias, min=-grad_norm, max=grad_norm).size())
                        # print(pretrain_model["_classifier.net.{}.bias".format(index * 2)].size())
                        # print(pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(1024, 1, 1).permute(0, 2, 1).size())
                        # weight = torch.clip(weight, min=-grad_norm, max=grad_norm) + pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1)[:sample_num]
                        # bias = torch.clip(bias, min=-grad_norm, max=grad_norm) + pretrain_model["linear{}.bias".format(index + 3)].squeeze(1).squeeze(1)[:sample_num]
                        # print(torch.clip(weight, min=-grad_norm, max=grad_norm).size())
                        # print(weight.size())
                        # print(torch.clip(bias, min=-grad_norm, max=grad_norm).size())
                        # print(bias.size())
                        # print(pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(1024, 1, 1).permute(0, 2, 1).size())

                        weight = torch.clip(weight, min=-grad_norm, max=grad_norm) + pretrain_model[
                            "_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1)
                        bias = torch.clip(bias, min=-grad_norm, max=grad_norm) + pretrain_model[
                            "_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)
                        # weight = weight + pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1,
                        #                                                                                        1).permute(0, 2,
                        #                                                                                                   1)
                        # bias = bias + pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)

                    # x = self.activation_fns[index](torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias) if self.activation_fns[index] is not None else (torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias)
                    x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
                    x = self.modules[index](x) if index < len(self.modules) else x

                if return_grad:
                    return x, grad_list
                else:
                    return x


class HyperNetwork_FC_grad_no_clip(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC_grad_no_clip, self).__init__()

        self.modules = []
        units = [in_dimension] + list(units)
        # modules = []
        self.activation_fns = activation_fns
        # units = [in_dimension] + list(units)
        units = list(units)
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)
        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())

    # def forward(self, x, z_list, sample_num=32, pretrain_model=None, grad_norm=1):
    def forward(self, x, z, sample_num=1024, trigger_seq_length=30, pretrain_model=None, grad_norm=1, return_grad=False):
        units = self.units
        # z = z.view(sample_num, -1)

        user_state, _ = self._gru_cell(z)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        z = self._mlp_trans(user_state)
        # print(z.size())
        grad_list = []
        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            # z = z_list[index]
            # import ipdb
            # ipdb.set_trace()
            weight = torch.matmul(z, self.w1[index]) + self.b1[index]
            # grad_list.append(torch.clip(weight, min=-grad_norm, max=grad_norm))
            grad_list.append(weight)

            output_size = units[i]
            # print("=" * 50)
            # print(index)
            # print(self.units)
            # print(input_size)
            # print(output_size)
            # print(z.size())
            # print(self.w1[index].size())
            # print(self.b1[index].size())
            # print(weight.size())
            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)

            bias = torch.matmul(z, self.w2[index]) + self.b2[index]
            if pretrain_model is not None:
                # print("=" * 50)
                # print(torch.clip(weight, min=-0.01, max=0.01).size())
                # print(pretrain_model["linear{}.weight".format(index + 3)].size())
                # print(pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1).size())
                # print(pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1)[:sample_num].size())
                # print("*" * 50)
                # print(torch.clip(weight, min=-grad_norm, max=grad_norm).size())
                # print(pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(1024, 1, 1).permute(0, 2, 1).size())
                # print("-" * 50)
                # print(torch.clip(bias, min=-grad_norm, max=grad_norm).size())
                # print(pretrain_model["_classifier.net.{}.bias".format(index * 2)].size())
                # print(pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(1024, 1, 1).permute(0, 2, 1).size())
                # weight = torch.clip(weight, min=-grad_norm, max=grad_norm) + pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1)[:sample_num]
                # bias = torch.clip(bias, min=-grad_norm, max=grad_norm) + pretrain_model["linear{}.bias".format(index + 3)].squeeze(1).squeeze(1)[:sample_num]
                # print(torch.clip(weight, min=-grad_norm, max=grad_norm).size())
                # print(weight.size())
                # print(torch.clip(bias, min=-grad_norm, max=grad_norm).size())
                # print(bias.size())
                # print(pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(1024, 1, 1).permute(0, 2, 1).size())

                # weight = torch.clip(weight, min=-grad_norm, max=grad_norm) + pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1)
                # bias = torch.clip(bias, min=-grad_norm, max=grad_norm) + pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)
                weight = weight + pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1)
                bias = bias + pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)

            # x = self.activation_fns[index](torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias) if self.activation_fns[index] is not None else (torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias)
            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
            x = self.modules[index](x) if index < len(self.modules) else x

        if return_grad:
            return x, grad_list
        else:
            return x


class HyperNetwork_FC_grad_no_clip_gru(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC_grad_no_clip_gru, self).__init__()

        self.modules = []
        units = [in_dimension] + list(units)
        # modules = []
        self.activation_fns = activation_fns
        # units = [in_dimension] + list(units)
        units = list(units)
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self._mlp_trans_init = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )

        self._mlp_trans = nn.ModuleList([StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )] * len(units))

        self._mlp_trans2 = nn.ModuleList([StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )] * len(units))

        self._gru_cell_seq = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        initializer.default_weight_init(self._gru_cell_seq.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell_seq.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell_seq.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell_seq.bias_hh_l0)

        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            len(units) - 1,
            # batch_first=True
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)

        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())

    # def forward(self, x, z_list, sample_num=32, pretrain_model=None, grad_norm=1):
    def forward(self, x, z, sample_num=1024, trigger_seq_length=30, pretrain_model=None, grad_norm=1, return_grad=False):
        units = self.units
        # z = z.view(sample_num, -1)

        # user_state, _ = self._gru_cell(z)
        user_state, _ = self._gru_cell_seq(z)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        # z = self._mlp_trans(user_state)
        z = self._mlp_trans_init(user_state)
        # print(z.size())
        grad_list = []

        z_list = []
        for i in range(1, len(units)):
            # index = i - 1
            # z = self._mlp_trans[index](z)
            z_list.append(z)

        z_ = torch.stack(z_list)
        # _, z_gru = self._gru_cell(z.unsqueeze(0), z_)
        _, z_gru = self._gru_cell(z_)

        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            # z = z_list[index]
            # import ipdb
            # ipdb.set_trace()
            z = z_gru[index].squeeze(0)
            z = nn.Tanh()(self._mlp_trans2[index](z))
            grad_list.append(z)

            weight = torch.matmul(z, self.w1[index]) + self.b1[index]
            # grad_list.append(torch.clip(weight, min=-grad_norm, max=grad_norm))
            # grad_list.append(weight)

            output_size = units[i]
            # print("=" * 50)
            # print(index)
            # print(self.units)
            # print(input_size)
            # print(output_size)
            # print(z.size())
            # print(self.w1[index].size())
            # print(self.b1[index].size())
            # print(weight.size())
            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)

            bias = torch.matmul(z, self.w2[index]) + self.b2[index]
            if pretrain_model is not None:
                # print("=" * 50)
                # print(torch.clip(weight, min=-0.01, max=0.01).size())
                # print(pretrain_model["linear{}.weight".format(index + 3)].size())
                # print(pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1).size())
                # print(pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1)[:sample_num].size())
                # print("*" * 50)
                # print(torch.clip(weight, min=-grad_norm, max=grad_norm).size())
                # print(pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(1024, 1, 1).permute(0, 2, 1).size())
                # print("-" * 50)
                # print(torch.clip(bias, min=-grad_norm, max=grad_norm).size())
                # print(pretrain_model["_classifier.net.{}.bias".format(index * 2)].size())
                # print(pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(1024, 1, 1).permute(0, 2, 1).size())
                # weight = torch.clip(weight, min=-grad_norm, max=grad_norm) + pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1)[:sample_num]
                # bias = torch.clip(bias, min=-grad_norm, max=grad_norm) + pretrain_model["linear{}.bias".format(index + 3)].squeeze(1).squeeze(1)[:sample_num]
                # print(torch.clip(weight, min=-grad_norm, max=grad_norm).size())
                # print(weight.size())
                # print(torch.clip(bias, min=-grad_norm, max=grad_norm).size())
                # print(bias.size())
                # print(pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(1024, 1, 1).permute(0, 2, 1).size())

                # weight = torch.clip(weight, min=-grad_norm, max=grad_norm) + pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1)
                # bias = torch.clip(bias, min=-grad_norm, max=grad_norm) + pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)
                weight = weight + pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1)
                bias = bias + pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)

            # x = self.activation_fns[index](torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias) if self.activation_fns[index] is not None else (torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias)
            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
            x = self.modules[index](x) if index < len(self.modules) else x

        if return_grad:
            return x, grad_list
        else:
            return x


class HyperNetwork_FC_grad_no_clip_transformer(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC_grad_no_clip_transformer, self).__init__()

        self.modules = []
        units = [in_dimension] + list(units)
        # modules = []
        self.activation_fns = activation_fns
        # units = [in_dimension] + list(units)
        units = list(units)
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self._mlp_trans_init = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        self._mlp_trans = nn.ModuleList([StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )] * len(units))

        self._mlp_trans2 = nn.ModuleList([StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )] * len(units))

        self._gru_cell_seq = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )

        initializer.default_weight_init(self._gru_cell_seq.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell_seq.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell_seq.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell_seq.bias_hh_l0)

        # self._transformer_cell_seq = nn.Transformer(
        #     nhead=4, num_encoder_layers=2
        # )
        self._transformer_cell_seq = nn.TransformerEncoderLayer(
            d_model=model_conf.id_dimension,
            nhead=model_conf.nhead,
            dim_feedforward=4 * model_conf.id_dimension,
            dropout=0
        )

        initializer.default_weight_init(self._transformer_cell_seq.self_attn.in_proj_weight)
        initializer.default_weight_init(self._transformer_cell_seq.self_attn.out_proj.weight)
        initializer.default_bias_init(self._transformer_cell_seq.self_attn.in_proj_bias)
        initializer.default_bias_init(self._transformer_cell_seq.self_attn.out_proj.bias)

        initializer.default_weight_init(self._transformer_cell_seq.linear1.weight)
        initializer.default_bias_init(self._transformer_cell_seq.linear1.bias)
        initializer.default_weight_init(self._transformer_cell_seq.linear2.weight)
        initializer.default_bias_init(self._transformer_cell_seq.linear2.bias)

        # self._transformer_cell = nn.Transformer(
        #     nhead=4,
        #     # num_encoder_layers=2
        # )
        self._transformer_cell = nn.TransformerEncoderLayer(
            d_model=model_conf.id_dimension,
            nhead=model_conf.nhead,
            dim_feedforward=4 * model_conf.id_dimension,
            dropout=0
        )

        initializer.default_weight_init(self._transformer_cell.self_attn.in_proj_weight)
        initializer.default_weight_init(self._transformer_cell.self_attn.out_proj.weight)
        initializer.default_bias_init(self._transformer_cell.self_attn.in_proj_bias)
        initializer.default_bias_init(self._transformer_cell.self_attn.out_proj.bias)

        initializer.default_weight_init(self._transformer_cell.linear1.weight)
        initializer.default_bias_init(self._transformer_cell.linear1.bias)
        initializer.default_weight_init(self._transformer_cell.linear2.weight)
        initializer.default_bias_init(self._transformer_cell.linear2.bias)
        # self._gru_cell = torch.nn.GRU(
        #     model_conf.id_dimension,
        #     model_conf.id_dimension,
        #     len(units) - 1,
        #     # batch_first=True
        # )
        # initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        # initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        # initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        # initializer.default_bias_init(self._gru_cell.bias_hh_l0)

        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())

    # def forward(self, x, z_list, sample_num=32, pretrain_model=None, grad_norm=1):
    def forward(self, x, z, sample_num=1024, trigger_seq_length=30, pretrain_model=None, grad_norm=1, return_grad=False):
        units = self.units
        # z = z.view(sample_num, -1)
        # print("=" * 50)
        # user_state, _ = self._gru_cell(z)
        # user_state, _ = self._gru_cell_seq(z)
        # print(user_state.size())
        user_state = self._transformer_cell_seq(z)
        # print(user_state.size())
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        # z = self._mlp_trans(user_state)
        z = self._mlp_trans_init(user_state)
        # print(z.size())
        grad_list = []

        z_list = []
        for i in range(1, len(units)):
            # index = i - 1
            # z = self._mlp_trans[index](z)
            z_list.append(z)

        z_ = torch.stack(z_list)

        # _, z_gru = self._gru_cell(z.unsqueeze(0), z_)
        # _, z_gru = self._gru_cell(z_)
        # z_gru = self._transformer_cell(z_)
        # print(z.unsqueeze(0).size())
        # print(z_.size())
        # z_gru = self._transformer_cell(z.unsqueeze(0), z_)
        z_gru = self._transformer_cell(z_)
        # print(z_gru.size())

        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            # z = z_list[index]
            # import ipdb
            # ipdb.set_trace()
            z = z_gru[index].squeeze(0)
            # z = self._mlp_trans2[index](z)
            grad_list.append(z)

            weight = torch.matmul(z, self.w1[index]) + self.b1[index]
            # grad_list.append(torch.clip(weight, min=-grad_norm, max=grad_norm))
            # grad_list.append(weight)

            output_size = units[i]
            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)

            bias = torch.matmul(z, self.w2[index]) + self.b2[index]
            if pretrain_model is not None:
                weight = weight + pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1)
                bias = bias + pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)

            # x = self.activation_fns[index](torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias) if self.activation_fns[index] is not None else (torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias)
            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
            x = self.modules[index](x) if index < len(self.modules) else x

        if return_grad:
            return x, grad_list
        else:
            return x


class HyperNetwork_FC_grad_no_clip_transformer_backup(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC_grad_no_clip_transformer_backup, self).__init__()

        self.modules = []
        units = [in_dimension] + list(units)
        # modules = []
        self.activation_fns = activation_fns
        # units = [in_dimension] + list(units)
        units = list(units)
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        self.output_size = []
        self._mlp_trans_init = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        self._mlp_trans = nn.ModuleList([StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )] * len(units))

        self._mlp_trans = nn.ModuleList([StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )] * len(units))

        self._mlp_trans2 = nn.ModuleList([StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )] * len(units))

        self._gru_cell_seq = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )

        initializer.default_weight_init(self._gru_cell_seq.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell_seq.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell_seq.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell_seq.bias_hh_l0)

        # self._transformer_cell_seq = nn.Transformer(
        #     nhead=4, num_encoder_layers=2
        # )
        self._transformer_cell_seq = nn.TransformerEncoderLayer(
            d_model=model_conf.id_dimension,
            nhead=model_conf.nhead,
            dim_feedforward=4 * model_conf.id_dimension,
            dropout=0
        )

        initializer.default_weight_init(self._transformer_cell_seq.self_attn.in_proj_weight)
        initializer.default_weight_init(self._transformer_cell_seq.self_attn.out_proj.weight)
        initializer.default_bias_init(self._transformer_cell_seq.self_attn.in_proj_bias)
        initializer.default_bias_init(self._transformer_cell_seq.self_attn.out_proj.bias)

        initializer.default_weight_init(self._transformer_cell_seq.linear1.weight)
        initializer.default_bias_init(self._transformer_cell_seq.linear1.bias)
        initializer.default_weight_init(self._transformer_cell_seq.linear2.weight)
        initializer.default_bias_init(self._transformer_cell_seq.linear2.bias)

        # self._transformer_cell = nn.Transformer(
        #     nhead=4,
        #     # num_encoder_layers=2
        # )
        self._transformer_cell = nn.TransformerEncoderLayer(
            d_model=model_conf.id_dimension,
            nhead=model_conf.nhead,
            dim_feedforward=4 * model_conf.id_dimension,
            dropout=0
        )

        initializer.default_weight_init(self._transformer_cell.self_attn.in_proj_weight)
        initializer.default_weight_init(self._transformer_cell.self_attn.out_proj.weight)
        initializer.default_bias_init(self._transformer_cell.self_attn.in_proj_bias)
        initializer.default_bias_init(self._transformer_cell.self_attn.out_proj.bias)

        initializer.default_weight_init(self._transformer_cell.linear1.weight)
        initializer.default_bias_init(self._transformer_cell.linear1.bias)
        initializer.default_weight_init(self._transformer_cell.linear2.weight)
        initializer.default_bias_init(self._transformer_cell.linear2.bias)
        # self._gru_cell = torch.nn.GRU(
        #     model_conf.id_dimension,
        #     model_conf.id_dimension,
        #     len(units) - 1,
        #     # batch_first=True
        # )
        # initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        # initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        # initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        # initializer.default_bias_init(self._gru_cell.bias_hh_l0)

        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())

    # def forward(self, x, z_list, sample_num=32, pretrain_model=None, grad_norm=1):
    def forward(self, x, z, sample_num=1024, trigger_seq_length=30, pretrain_model=None, grad_norm=1, return_grad=False):
        units = self.units
        # z = z.view(sample_num, -1)
        # print("=" * 50)
        # user_state, _ = self._gru_cell(z)
        user_state, _ = self._gru_cell_seq(z)
        # print(user_state.size())
        user_state = self._transformer_cell_seq(z)
        # print(user_state.size())
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        # z = self._mlp_trans(user_state)
        z = self._mlp_trans_init(user_state)
        # print(z.size())
        grad_list = []

        z_list = []
        for i in range(1, len(units)):
            index = i - 1
            z = self._mlp_trans[index](z)
            z_list.append(z)

        z_ = torch.stack(z_list)
        # _, z_gru = self._gru_cell(z.unsqueeze(0), z_)
        # _, z_gru = self._gru_cell(z_)
        # z_gru = self._transformer_cell(z_)
        # print(z.unsqueeze(0).size())
        # print(z_.size())
        # z_gru = self._transformer_cell(z.unsqueeze(0), z_)
        z_gru = self._transformer_cell(z_)
        # print(z_gru.size())

        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            # z = z_list[index]
            # import ipdb
            # ipdb.set_trace()
            z = z_gru[index].squeeze(0)
            z = self._mlp_trans2[index](z)
            grad_list.append(z)

            weight = torch.matmul(z, self.w1[index]) + self.b1[index]
            # grad_list.append(torch.clip(weight, min=-grad_norm, max=grad_norm))
            # grad_list.append(weight)

            output_size = units[i]
            # print("=" * 50)
            # print(index)
            # print(self.units)
            # print(input_size)
            # print(output_size)
            # print(z.size())
            # print(self.w1[index].size())
            # print(self.b1[index].size())
            # print(weight.size())
            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)

            bias = torch.matmul(z, self.w2[index]) + self.b2[index]
            if pretrain_model is not None:
                # print("=" * 50)
                # print(torch.clip(weight, min=-0.01, max=0.01).size())
                # print(pretrain_model["linear{}.weight".format(index + 3)].size())
                # print(pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1).size())
                # print(pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1)[:sample_num].size())
                # print("*" * 50)
                # print(torch.clip(weight, min=-grad_norm, max=grad_norm).size())
                # print(pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(1024, 1, 1).permute(0, 2, 1).size())
                # print("-" * 50)
                # print(torch.clip(bias, min=-grad_norm, max=grad_norm).size())
                # print(pretrain_model["_classifier.net.{}.bias".format(index * 2)].size())
                # print(pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(1024, 1, 1).permute(0, 2, 1).size())
                # weight = torch.clip(weight, min=-grad_norm, max=grad_norm) + pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1)[:sample_num]
                # bias = torch.clip(bias, min=-grad_norm, max=grad_norm) + pretrain_model["linear{}.bias".format(index + 3)].squeeze(1).squeeze(1)[:sample_num]
                # print(torch.clip(weight, min=-grad_norm, max=grad_norm).size())
                # print(weight.size())
                # print(torch.clip(bias, min=-grad_norm, max=grad_norm).size())
                # print(bias.size())
                # print(pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(1024, 1, 1).permute(0, 2, 1).size())

                # weight = torch.clip(weight, min=-grad_norm, max=grad_norm) + pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1)
                # bias = torch.clip(bias, min=-grad_norm, max=grad_norm) + pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)
                weight = weight + pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1)
                bias = bias + pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)

            # x = self.activation_fns[index](torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias) if self.activation_fns[index] is not None else (torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias)
            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
            x = self.modules[index](x) if index < len(self.modules) else x

        if return_grad:
            return x, grad_list
        else:
            return x


# class HyperNetwork_FC_grad_no_clip_gru2(nn.Module):
#     # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
#     def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
#                  model_conf=None, expand=False):
#         super(HyperNetwork_FC_grad_no_clip_gru2, self).__init__()
#
#         self.modules = []
#         units = [in_dimension] + list(units)
#         # modules = []
#         self.activation_fns = activation_fns
#         # units = [in_dimension] + list(units)
#         units = list(units)
#         self.batch = batch
#         self.units = units
#         self.w1 = []
#         self.w2 = []
#         self.b1 = []
#         self.b2 = []
#         self.output_size = []
#         self._mlp_trans_init = StackedDense(
#             model_conf.id_dimension,
#             [model_conf.id_dimension] * model_conf.mlp_layers,
#             ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
#         )
#         self._mlp_trans = nn.ModuleList([StackedDense(
#             model_conf.id_dimension,
#             [model_conf.id_dimension] * model_conf.mlp_layers,
#             ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
#         )] * len(units))
#
#         self._mlp_trans = nn.ModuleList([StackedDense(
#             model_conf.id_dimension,
#             [model_conf.id_dimension] * model_conf.mlp_layers,
#             ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
#         )] * len(units))
#
#         self._mlp_trans2 = nn.ModuleList([StackedDense(
#             model_conf.id_dimension,
#             [model_conf.id_dimension] * model_conf.mlp_layers,
#             ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
#         )] * len(units))
#
#         self._gru_cell_seq = torch.nn.GRU(
#             model_conf.id_dimension,
#             model_conf.id_dimension,
#             batch_first=True
#         )
#         initializer.default_weight_init(self._gru_cell_seq.weight_hh_l0)
#         initializer.default_weight_init(self._gru_cell_seq.weight_ih_l0)
#         initializer.default_bias_init(self._gru_cell_seq.bias_ih_l0)
#         initializer.default_bias_init(self._gru_cell_seq.bias_hh_l0)
#
#         self._gru_cell = torch.nn.GRU(
#             model_conf.id_dimension,
#             model_conf.id_dimension,
#             len(units) - 1,
#             # batch_first=True
#         )
#         initializer.default_weight_init(self._gru_cell.weight_hh_l0)
#         initializer.default_weight_init(self._gru_cell.weight_ih_l0)
#         initializer.default_bias_init(self._gru_cell.bias_ih_l0)
#         initializer.default_bias_init(self._gru_cell.bias_hh_l0)
#
#         self.expand = expand
#         for i in range(1, len(units)):
#             if i == 1 and self.expand:
#                 input_size = units[i - 1] * 2
#             else:
#                 input_size = units[i - 1]
#
#             output_size = units[i]
#
#             self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
#             self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))
#
#             self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
#             self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))
#
#             if activation_fns[i - 1] is not None:
#                 self.modules.append(activation_fns[i - 1]())
#
#     # def forward(self, x, z_list, sample_num=32, pretrain_model=None, grad_norm=1):
#     def forward(self, x, z, sample_num=1024, trigger_seq_length=30, pretrain_model=None, grad_norm=1, return_grad=False):
#         units = self.units
#         # z = z.view(sample_num, -1)
#
#         # user_state, _ = self._gru_cell(z)
#         user_state, _ = self._gru_cell_seq(z)
#         user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
#         # z = self._mlp_trans(user_state)
#         z = self._mlp_trans_init(user_state)
#         # print(z.size())
#         grad_list = []
#
#         z_list = []
#         for i in range(1, len(units)):
#             index = i - 1
#             z = self._mlp_trans[index](z)
#             z_list.append(z)
#
#         z_ = torch.stack(z_list)
#         # _, z_gru = self._gru_cell(z.unsqueeze(0), z_)
#         _, z_gru = self._gru_cell(z_)
#
#         for i in range(1, len(units)):
#             index = i - 1
#             if i == 1 and self.expand:
#                 input_size = units[i - 1] * 2
#             else:
#                 input_size = units[i - 1]
#             # z = z_list[index]
#             # import ipdb
#             # ipdb.set_trace()
#             z = z_gru[index].squeeze(0)
#             z = self._mlp_trans2[index](z)
#             grad_list.append(z)
#
#             weight = torch.matmul(z, self.w1[index]) + self.b1[index]
#             # grad_list.append(torch.clip(weight, min=-grad_norm, max=grad_norm))
#             # grad_list.append(weight)
#
#             output_size = units[i]
#             if not self.batch:
#                 weight = weight.view(input_size, output_size)
#             else:
#                 weight = weight.view(sample_num, input_size, output_size)
#
#             bias = torch.matmul(z, self.w2[index]) + self.b2[index]
#             if pretrain_model is not None:
#                 weight = weight + pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1)
#                 bias = bias + pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)
#
#             # x = self.activation_fns[index](torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias) if self.activation_fns[index] is not None else (torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias)
#             x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
#             x = self.modules[index](x) if index < len(self.modules) else x
#
#         if return_grad:
#             return x, grad_list
#         else:
#             return x

class HyperNetwork_FC_grad_2(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC_grad_2, self).__init__()

        self.modules = []
        units = [in_dimension] + list(units)
        # modules = []
        self.activation_fns = activation_fns
        # units = [in_dimension] + list(units)
        units = list(units)
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        # self.w_aggregator = []
        # self.w_aggregator_activation = []
        # self.b_aggregator = []
        # self.b_aggregator_activation = []
        self.w_aggregator = nn.ModuleList()
        self.w_aggregator_activation = nn.ModuleList()
        self.b_aggregator = nn.ModuleList()
        self.b_aggregator_activation = nn.ModuleList()
        self.output_size = []
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)

        # self._mlp_trans = StackedDense(
        #     model_conf.id_dimension,
        #     [model_conf.id_dimension] * model_conf.mlp_layers,
        #     ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        # )

        self._mlp_trans_init = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )

        self._mlp_trans = nn.ModuleList([StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )] * len(units))

        self._mlp_trans2 = nn.ModuleList([StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )] * len(units))

        self.expand = expand
        self._mlp_trans_param = nn.ModuleList()
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            # self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension * 2, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            # self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension * 2, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            # self.w_aggregator.append(
            #     nn.Linear(input_size * output_size * 2, input_size * output_size)
            # )
            # # self.w_aggregator_activation.append(nn.ReLU)
            # self.w_aggregator_activation.append(nn.Tanh())
            # self.b_aggregator.append(
            #     nn.Linear(input_size * output_size * 2, input_size * output_size)
            # )
            # # self.b_aggregator_activation.append(nn.ReLU)
            # self.b_aggregator_activation.append(nn.Tanh())

            self._mlp_trans_param.append(
                StackedDense(
                    # model_conf.id_dimension,
                    in_dimension * output_size,
                    [model_conf.id_dimension],
                    [torch.nn.Tanh]
                )
            )
            # self._mlp_trans2 = nn.ModuleList([StackedDense(
            #     model_conf.id_dimension,
            #     [model_conf.id_dimension] * model_conf.mlp_layers,
            #     ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
            # )] * len(units))


            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())


    # def forward(self, x, z_list, sample_num=32, pretrain_model=None, grad_norm=1):
    def forward(self, x, z, sample_num=1024, trigger_seq_length=30, pretrain_model=None, grad_norm=1, return_grad=False):
        units = self.units
        # z = z.view(sample_num, -1)

        user_state, _ = self._gru_cell(z)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        # z = self._mlp_trans(user_state)
        z = self._mlp_trans_init(user_state)
        # print(z.size())
        grad_list = []
        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            # z = z_list[index]
            # import ipdb
            # ipdb.set_trace()

            # weight = torch.matmul(z, self.w1[index]) + self.b1[index]

            if pretrain_model is not None:
                # self.w_aggregator[index].cuda()
                # print("=" * 50)
                # print(weight.size())
                pretrain_model_feature = self._mlp_trans_param[index](pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).\
                    permute(0, 2, 1).reshape(sample_num, -1))
                # print(pretrain_model_feature.size())
                # print(z.size())
                # print(torch.cat([z, pretrain_model_feature], dim=1).size())
                weight = torch.matmul(torch.cat([z, pretrain_model_feature], dim=1), self.w1[index]) + self.b1[index]
                bias = torch.matmul(torch.cat([z, pretrain_model_feature], dim=1), self.w2[index]) + self.b2[index]
                # weight = self.w_aggregator[index](
                #     # torch.cat([weight, pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1).reshape(sample_num, -1)], dim=1)
                #     torch.cat([weight, pretrain_model_feature], dim=1)
                # )
                # print(weight.size())
                grad_list.append(torch.clip(weight, min=-grad_norm, max=grad_norm))
            output_size = units[i]

            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)

            # bias = torch.matmul(z, self.w2[index]) + self.b2[index]

            if pretrain_model is not None:

                # self.w_aggregator[index].cuda()
                # weight = self.w_aggregator[index](
                #     torch.cat([weight, pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1).reshape(sample_num, -1)], dim=1)
                # )

                weight = torch.clip(weight, min=-grad_norm, max=grad_norm) + pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1)
                bias = torch.clip(bias, min=-grad_norm, max=grad_norm) + pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)

            # x = self.activation_fns[index](torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias) if self.activation_fns[index] is not None else (torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias)
            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
            x = self.modules[index](x) if index < len(self.modules) else x

        if return_grad:
            return x, grad_list
        else:
            return x


class HyperNetwork_FC_grad_2_no_clip(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC_grad_2_no_clip, self).__init__()

        self.modules = []
        units = [in_dimension] + list(units)
        # modules = []
        self.activation_fns = activation_fns
        # units = [in_dimension] + list(units)
        units = list(units)
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        # self.w_aggregator = []
        # self.w_aggregator_activation = []
        # self.b_aggregator = []
        # self.b_aggregator_activation = []
        self.w_aggregator = nn.ModuleList()
        self.w_aggregator_activation = nn.ModuleList()
        self.b_aggregator = nn.ModuleList()
        self.b_aggregator_activation = nn.ModuleList()
        self.output_size = []
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)
        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            self.w_aggregator.append(
                nn.Linear(input_size * output_size * 2, input_size * output_size)
            )
            # self.w_aggregator_activation.append(nn.ReLU)
            self.w_aggregator_activation.append(nn.Tanh())
            self.b_aggregator.append(
                nn.Linear(input_size * output_size * 2, input_size * output_size)
            )
            # self.b_aggregator_activation.append(nn.ReLU)
            self.b_aggregator_activation.append(nn.Tanh())

            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())

    # def forward(self, x, z_list, sample_num=32, pretrain_model=None, grad_norm=1):
    def forward(self, x, z, sample_num=1024, trigger_seq_length=30, pretrain_model=None, grad_norm=1, return_grad=False):
        units = self.units
        # z = z.view(sample_num, -1)

        user_state, _ = self._gru_cell(z)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        z = self._mlp_trans(user_state)
        # print(z.size())
        grad_list = []
        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            # z = z_list[index]
            # import ipdb
            # ipdb.set_trace()
            weight = torch.matmul(z, self.w1[index]) + self.b1[index]

            if pretrain_model is not None:
                # self.w_aggregator[index].cuda()
                # print("=" * 50)
                # print(weight.size())
                weight = self.w_aggregator[index](
                    torch.cat([weight, pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1).reshape(sample_num, -1)], dim=1)
                )
                # print(weight.size())
                # grad_list.append(torch.clip(weight, min=-grad_norm, max=grad_norm))
                grad_list.append(weight)
            output_size = units[i]

            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)

            bias = torch.matmul(z, self.w2[index]) + self.b2[index]
            if pretrain_model is not None:

                # self.w_aggregator[index].cuda()
                # weight = self.w_aggregator[index](
                #     torch.cat([weight, pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1).reshape(sample_num, -1)], dim=1)
                # )

                # weight = torch.clip(weight, min=-grad_norm, max=grad_norm) + pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1)
                # bias = torch.clip(bias, min=-grad_norm, max=grad_norm) + pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)
                weight = weight + pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1)
                bias = bias + pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)

            # x = self.activation_fns[index](torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias) if self.activation_fns[index] is not None else (torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias)
            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
            x = self.modules[index](x) if index < len(self.modules) else x

        if return_grad:
            return x, grad_list
        else:
            return x


class HyperNetwork_FC_grad_2_no_clip_gru(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC_grad_2_no_clip_gru, self).__init__()

        self.modules = []
        units = [in_dimension] + list(units)
        # modules = []
        self.activation_fns = activation_fns
        # units = [in_dimension] + list(units)
        units = list(units)
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        # self.w_aggregator = []
        # self.w_aggregator_activation = []
        # self.b_aggregator = []
        # self.b_aggregator_activation = []
        self.w_aggregator = nn.ModuleList()
        self.w_aggregator_activation = nn.ModuleList()
        self.b_aggregator = nn.ModuleList()
        self.b_aggregator_activation = nn.ModuleList()
        self.output_size = []
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)
        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            self.w_aggregator.append(
                nn.Linear(input_size * output_size * 2, input_size * output_size)
            )
            # self.w_aggregator_activation.append(nn.ReLU)
            self.w_aggregator_activation.append(nn.Tanh())
            self.b_aggregator.append(
                nn.Linear(input_size * output_size * 2, input_size * output_size)
            )
            # self.b_aggregator_activation.append(nn.ReLU)
            self.b_aggregator_activation.append(nn.Tanh())

            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())

    # def forward(self, x, z_list, sample_num=32, pretrain_model=None, grad_norm=1):
    def forward(self, x, z, sample_num=1024, trigger_seq_length=30, pretrain_model=None, grad_norm=1, return_grad=False):
        units = self.units
        # z = z.view(sample_num, -1)

        user_state, _ = self._gru_cell(z)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        z = self._mlp_trans(user_state)
        # print(z.size())
        grad_list = []
        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            # z = z_list[index]
            # import ipdb
            # ipdb.set_trace()
            weight = torch.matmul(z, self.w1[index]) + self.b1[index]

            if pretrain_model is not None:
                # self.w_aggregator[index].cuda()
                # print("=" * 50)
                # print(weight.size())
                weight = self.w_aggregator[index](
                    torch.cat([weight, pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1).reshape(sample_num, -1)], dim=1)
                )
                # print(weight.size())
                # grad_list.append(torch.clip(weight, min=-grad_norm, max=grad_norm))
                grad_list.append(weight)
            output_size = units[i]

            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)

            bias = torch.matmul(z, self.w2[index]) + self.b2[index]
            if pretrain_model is not None:

                # self.w_aggregator[index].cuda()
                # weight = self.w_aggregator[index](
                #     torch.cat([weight, pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1).reshape(sample_num, -1)], dim=1)
                # )

                # weight = torch.clip(weight, min=-grad_norm, max=grad_norm) + pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1)
                # bias = torch.clip(bias, min=-grad_norm, max=grad_norm) + pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)
                weight = weight + pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1)
                bias = bias + pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)

            # x = self.activation_fns[index](torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias) if self.activation_fns[index] is not None else (torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias)
            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
            x = self.modules[index](x) if index < len(self.modules) else x

        if return_grad:
            return x, grad_list
        else:
            return x


# class HyperNetwork_FC_grad_2_no_clip_gru(nn.Module):
#     # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
#     def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
#                  model_conf=None, expand=False):
#         super(HyperNetwork_FC_grad_2_no_clip_gru, self).__init__()
#
#         self.modules = []
#         units = [in_dimension] + list(units)
#         # modules = []
#         self.activation_fns = activation_fns
#         # units = [in_dimension] + list(units)
#         units = list(units)
#         self.batch = batch
#         self.units = units
#         self.w1 = []
#         self.w2 = []
#         self.b1 = []
#         self.b2 = []
#         # self.w_aggregator = []
#         # self.w_aggregator_activation = []
#         # self.b_aggregator = []
#         # self.b_aggregator_activation = []
#         self.w_aggregator = nn.ModuleList()
#         self.w_aggregator_activation = nn.ModuleList()
#         self.b_aggregator = nn.ModuleList()
#         self.b_aggregator_activation = nn.ModuleList()
#         self.output_size = []
#
#         self._mlp_trans_init = StackedDense(
#             model_conf.id_dimension,
#             [model_conf.id_dimension] * model_conf.mlp_layers,
#             ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
#         )
#         # self._mlp_trans =
#         self._mlp_trans = nn.ModuleList([StackedDense(
#             model_conf.id_dimension,
#             [model_conf.id_dimension] * model_conf.mlp_layers,
#             ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
#         )] * len(units))
#
#         self._gru_cell_seq = torch.nn.GRU(
#             model_conf.id_dimension,
#             model_conf.id_dimension,
#             batch_first=True
#         )
#         initializer.default_weight_init(self._gru_cell_seq.weight_hh_l0)
#         initializer.default_weight_init(self._gru_cell_seq.weight_ih_l0)
#         initializer.default_bias_init(self._gru_cell_seq.bias_ih_l0)
#         initializer.default_bias_init(self._gru_cell_seq.bias_hh_l0)
#
#         self._gru_cell = torch.nn.GRU(
#             model_conf.id_dimension,
#             model_conf.id_dimension,
#             len(units) - 1,
#             # batch_first=True
#         )
#         initializer.default_weight_init(self._gru_cell.weight_hh_l0)
#         initializer.default_weight_init(self._gru_cell.weight_ih_l0)
#         initializer.default_bias_init(self._gru_cell.bias_ih_l0)
#         initializer.default_bias_init(self._gru_cell.bias_hh_l0)
#
#         self.expand = expand
#         for i in range(1, len(units)):
#             if i == 1 and self.expand:
#                 input_size = units[i - 1] * 2
#             else:
#                 input_size = units[i - 1]
#
#             output_size = units[i]
#
#             self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
#             self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))
#
#             self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
#             self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))
#
#             self.w_aggregator.append(
#                 nn.Linear(input_size * output_size * 2, input_size * output_size)
#             )
#             # self.w_aggregator_activation.append(nn.ReLU)
#             self.w_aggregator_activation.append(nn.Tanh())
#             self.b_aggregator.append(
#                 nn.Linear(input_size * output_size * 2, input_size * output_size)
#             )
#             # self.b_aggregator_activation.append(nn.ReLU)
#             self.b_aggregator_activation.append(nn.Tanh())
#
#             if activation_fns[i - 1] is not None:
#                 self.modules.append(activation_fns[i - 1]())
#
#     # def forward(self, x, z_list, sample_num=32, pretrain_model=None, grad_norm=1):
#     def forward(self, x, z, sample_num=1024, trigger_seq_length=30, pretrain_model=None, grad_norm=1, return_grad=False):
#         units = self.units
#         # z = z.view(sample_num, -1)
#
#         # user_state, _ = self._gru_cell(z)
#         user_state, _ = self._gru_cell_seq(z)
#         # print("=" * 50)
#         # print("user_state.size() ", user_state.size())
#         user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
#         # print("user_state.size() ", user_state.size())
#         # z = self._mlp_trans(user_state)
#         z = self._mlp_trans_init(user_state)
#         # from copy import deepcopy
#         # z_backup = deepcopy(z)
#         # z_backup = z
#         # print("z.size() ", z.size())
#         # print(z.size())
#         grad_list = []
#
#         # z_list = nn.ModuleList()
#         z_list = []
#         for i in range(1, len(units)):
#             index = i - 1
#             z = self._mlp_trans[index](z)
#             z_list.append(z)
#
#         z_ = torch.stack(z_list)
#         # print(z_.size())
#         # print(z.unsqueeze(0).size())
#         # print(z_.size())
#         # print(self._gru_cell)
#         # z_gru, _ = self._gru_cell(z.unsqueeze(0), z_)
#         _, z_gru = self._gru_cell(z.unsqueeze(0), z_)
#         # z_gru, _ = self._gru_cell(z.unsqueeze(1), z_)
#         # print(z_gru.size())
#         # print(_.size())
#         # print("z_.size() ", z_.size())
#
#         for i in range(1, len(units)):
#             index = i - 1
#             if i == 1 and self.expand:
#                 input_size = units[i - 1] * 2
#             else:
#                 input_size = units[i - 1]
#             # z = z_list[index]
#             # import ipdb
#             # ipdb.set_trace()
#
#             # if i == 1:
#             #     user_state, _ = self._gru_cell(z)
#             # else:
#             #     user_state, _ = self._gru_cell(user_state)
#             # z = z_[:, index, :].squeeze(1)
#             # z = z_[index, :, :].squeeze(0)
#
#             # print("-" * 50)
#             # z, _ = self._gru_cell(z)
#             # print("z.size() ", z.size())
#             # z = self._mlp_trans[index](z)
#
#             # print("z.size() ", z.size())
#             # z = self._gru_cell(z.unsquuze(0), z_backup.unsquuze(0)).squuze(0)
#             z = z_gru[index].squeeze(0)
#             grad_list.append(z)
#             weight = torch.matmul(z, self.w1[index]) + self.b1[index]
#
#             if pretrain_model is not None:
#                 # self.w_aggregator[index].cuda()
#                 # print("=" * 50)
#                 # print(weight.size())
#                 weight = self.w_aggregator[index](
#                     torch.cat([weight, pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1).reshape(sample_num, -1)], dim=1)
#                 )
#                 # print(weight.size())
#                 # grad_list.append(torch.clip(weight, min=-grad_norm, max=grad_norm))
#                 # grad_list.append(weight)
#             output_size = units[i]
#
#             if not self.batch:
#                 weight = weight.view(input_size, output_size)
#             else:
#                 weight = weight.view(sample_num, input_size, output_size)
#
#             bias = torch.matmul(z, self.w2[index]) + self.b2[index]
#             if pretrain_model is not None:
#
#                 # self.w_aggregator[index].cuda()
#                 # weight = self.w_aggregator[index](
#                 #     torch.cat([weight, pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1).reshape(sample_num, -1)], dim=1)
#                 # )
#
#                 # weight = torch.clip(weight, min=-grad_norm, max=grad_norm) + pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1)
#                 # bias = torch.clip(bias, min=-grad_norm, max=grad_norm) + pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)
#                 weight = weight + pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1)
#                 bias = bias + pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1)
#
#             # x = self.activation_fns[index](torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias) if self.activation_fns[index] is not None else (torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias)
#             x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
#             x = self.modules[index](x) if index < len(self.modules) else x
#
#         if return_grad:
#             return x, grad_list
#         else:
#             return x



class HyperNetwork_FC_param(nn.Module):
    # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
    def __init__(self, in_dimension, units, activation_fns, batch=True, trigger_sequence_len=30,
                 model_conf=None, expand=False):
        super(HyperNetwork_FC_param, self).__init__()

        self.modules = []
        units = [in_dimension] + list(units)
        # modules = []
        self.activation_fns = activation_fns
        # units = [in_dimension] + list(units)
        units = list(units)
        self.batch = batch
        self.units = units
        self.w1 = []
        self.w2 = []
        self.b1 = []
        self.b2 = []
        # self.w_aggregator = []
        # self.w_aggregator_activation = []
        # self.b_aggregator = []
        # self.b_aggregator_activation = []
        self.w_aggregator = nn.ModuleList()
        self.w_aggregator_activation = nn.ModuleList()
        self.b_aggregator = nn.ModuleList()
        self.b_aggregator_activation = nn.ModuleList()
        self.output_size = []
        self._gru_cell = torch.nn.GRU(
            model_conf.id_dimension,
            model_conf.id_dimension,
            batch_first=True
        )
        self._mlp_trans = StackedDense(
            model_conf.id_dimension,
            [model_conf.id_dimension] * model_conf.mlp_layers,
            ([torch.nn.Tanh] * (model_conf.mlp_layers - 1)) + [None]
        )
        initializer.default_weight_init(self._gru_cell.weight_hh_l0)
        initializer.default_weight_init(self._gru_cell.weight_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_ih_l0)
        initializer.default_bias_init(self._gru_cell.bias_hh_l0)
        self.expand = expand
        for i in range(1, len(units)):
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]

            output_size = units[i]

            self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
            self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))

            self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
            self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))

            self.w_aggregator.append(
                nn.Linear(input_size * output_size * 2, input_size * output_size)
            )
            # self.w_aggregator_activation.append(nn.ReLU)
            self.w_aggregator_activation.append(nn.Tanh())
            self.b_aggregator.append(
                nn.Linear(input_size * output_size * 2, input_size * output_size)
            )
            # self.b_aggregator_activation.append(nn.ReLU)
            self.b_aggregator_activation.append(nn.Tanh())
            if activation_fns[i - 1] is not None:
                self.modules.append(activation_fns[i - 1]())

    # def forward(self, x, z_list, sample_num=32, pretrain_model=None, grad_norm=1):
    def forward(self, x, z, sample_num=1024, trigger_seq_length=30, pretrain_model=None, grad_norm=1):
        units = self.units
        # z = z.view(sample_num, -1)

        user_state, _ = self._gru_cell(z)
        user_state = user_state[range(user_state.shape[0]), trigger_seq_length, :]
        z = self._mlp_trans(user_state)
        # print(z.size())
        for i in range(1, len(units)):
            index = i - 1
            if i == 1 and self.expand:
                input_size = units[i - 1] * 2
            else:
                input_size = units[i - 1]
            # z = z_list[index]
            # import ipdb
            # ipdb.set_trace()
            weight = torch.matmul(z, self.w1[index]) + self.b1[index]
            if pretrain_model is not None:
                # print("=" * 50)
                # print(weight.size())
                # print(pretrain_model["_classifier.net.{}.weight".format(index * 2)].
                #                 repeat(sample_num, 1, 1).permute(0, 2, 1).size())
                # # pretrain_weight = pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1).view(sample_num, -1)
                # # pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2,
                # #                                                                                                1).view(
                # #     sample_num, -1)
                # print(pretrain_weight.size())
                # print(pretrain_model["_classifier.net.{}.weight".format(index * 2)].
                #                 repeat(sample_num, 1, 1).permute(0, 2, 1).view(sample_num, 1).size())
                # print(torch.cat([weight, pretrain_model["_classifier.net.{}.weight".format(index * 2)].
                #                 repeat(sample_num, 1, 1).permute(0, 2, 1).reshape(sample_num, -1)]).size())
                #                 # repeat(sample_num, 1, 1).permute(0, 2, 1).view(sample_num, -1)]).size())
                # weight = self.w_aggregator_activation[index](
                #     # self.w_aggregator[index](weight, pretrain_model["_classifier{}.weight".format(index)])
                #     # self.w_aggregator[index](torch.cat([weight, pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1)], dim=1))
                #     self.w_aggregator[index](torch.cat([weight, pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1).reshape(sample_num, -1)], dim=1))
                # )
                # self.w_aggregator[index].cuda()
                weight = self.w_aggregator[index](torch.cat([weight, pretrain_model["_classifier.net.{}.weight".format(index * 2)].repeat(sample_num, 1, 1).permute(0, 2, 1).reshape(sample_num, -1)], dim=1))
            output_size = units[i]
            if not self.batch:
                weight = weight.view(input_size, output_size)
            else:
                weight = weight.view(sample_num, input_size, output_size)

            bias = torch.matmul(z, self.w2[index]) + self.b2[index]
            # if pretrain_model is not None:
            #     # bias = self.b_aggregator_activation[index](
            #     #     # self.b_aggregator[index](weight, pretrain_model["_classifier{}.bias".format(index)])
            #     #     # self.b_aggregator[index](weight, pretrain_model["_classifier.net.{}.bias".format(index * 2)]).repeat(sample_num, 1)
            #     #     self.b_aggregator[index](torch.cat([bias, pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1).reshape(sample_num, -1)], dim=1))
            #     # )
            #     self.b_aggregator[index].cuda()
            #     print(bias.size())
            #     print(pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1).reshape(sample_num, -1).size())
            #     bias = self.b_aggregator[index](torch.cat([bias, pretrain_model["_classifier.net.{}.bias".format(index * 2)].repeat(sample_num, 1).reshape(sample_num, -1)], dim=1))

            # if pretrain_model is not None:
            #     # print("=" * 50)
            #     # print(torch.clip(weight, min=-0.01, max=0.01).size())
            #     # print(pretrain_model["linear{}.weight".format(index + 3)].size())
            #     # print(pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1).size())
            #     # print(pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1)[:sample_num].size())
            #     # print("-" * 50)
            #     # print(torch.clip(bias, min=-0.01, max=0.01).size())
            #     # print(pretrain_model["linear{}.bias".format(index + 3)].size())
            #
            #     # weight = torch.clip(weight, min=-grad_norm, max=grad_norm) + pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1)[:sample_num]
            #     # bias = torch.clip(bias, min=-grad_norm, max=grad_norm) + pretrain_model["linear{}.bias".format(index + 3)].squeeze(1).squeeze(1)[:sample_num]
            #     weight = torch.clip(weight, min=-grad_norm, max=grad_norm) + pretrain_model["_classifier{}.weight".format(index)]
            #     bias = torch.clip(bias, min=-grad_norm, max=grad_norm) + pretrain_model["_classifier{}.bias".format(index)]

            # x = self.activation_fns[index](torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias) if self.activation_fns[index] is not None else (torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias)
            x = torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
            # x = self.activation_fns[index]()(x)
            x = self.modules[index](x) if index < len(self.modules) else x
            
        return x


# class HyperNetwork_FC_cv_grad(nn.Module):
#     # def __init__(self, f_size=3, z_dim=64, out_size=16, in_size=16, batch=False):
#     def __init__(self, in_dimension, units, activation_fns, batch=True):
#         super(HyperNetwork_FC_grad, self).__init__()
#
#         # modules = []
#         self.activation_fns = activation_fns
#         # units = [in_dimension] + list(units)
#         units = list(units)
#         self.batch = batch
#         self.units = units
#         self.w1 = []
#         self.w2 = []
#         self.b1 = []
#         self.b2 = []
#         self.output_size = []
#
#         for i in range(1, len(units)):
#
#             input_size = units[i - 1]
#             output_size = units[i]
#
#             self.w1.append(Parameter(torch.fmod(torch.randn(in_dimension, input_size * output_size).cuda(), 2)))
#             self.b1.append(Parameter(torch.fmod(torch.randn(input_size * output_size).cuda(), 2)))
#
#             self.w2.append(Parameter(torch.fmod(torch.randn(in_dimension, output_size).cuda(), 2)))
#             self.b2.append(Parameter(torch.fmod(torch.randn(output_size).cuda(), 2)))
#
#             # if activation_fns[i - 1] is not None:
#             #     modules.append(activation_fns[i - 1]())
#
#     def forward(self, x, z_list, sample_num=32, pretrain_model=None, grad_norm=1):
#         units = self.units
#         # z = z.view(sample_num, -1)
#         # print(z.size())
#         for i in range(1, len(units)):
#             index = i - 1
#             input_size = units[i - 1]
#             z = z_list[index]
#             # import ipdb
#             # ipdb.set_trace()
#             weight = torch.matmul(z, self.w1[index]) + self.b1[index]
#
#             output_size = units[i]
#             if not self.batch:
#                 weight = weight.view(input_size, output_size)
#             else:
#                 weight = weight.view(sample_num, input_size, output_size)
#
#             bias = torch.matmul(z, self.w2[index]) + self.b2[index]
#             if pretrain_model is not None:
#                 # print("=" * 50)
#                 # print(torch.clip(weight, min=-0.01, max=0.01).size())
#                 # print(pretrain_model["linear{}.weight".format(index + 3)].size())
#                 # print(pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1).size())
#                 # print(pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1)[:sample_num].size())
#                 # print("-" * 50)
#                 # print(torch.clip(bias, min=-0.01, max=0.01).size())
#                 # print(pretrain_model["linear{}.bias".format(index + 3)].size())
#
#                 weight = torch.clip(weight, min=-grad_norm, max=grad_norm) + pretrain_model["linear{}.weight".format(index + 3)].squeeze(1).permute(0, 2, 1)[:sample_num]
#                 bias = torch.clip(bias, min=-grad_norm, max=grad_norm) + pretrain_model["linear{}.bias".format(index + 3)].squeeze(1).squeeze(1)[:sample_num]
#
#             x = self.activation_fns[index](torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias) if self.activation_fns[index] is not None else (torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias)
#
#         return x
#
