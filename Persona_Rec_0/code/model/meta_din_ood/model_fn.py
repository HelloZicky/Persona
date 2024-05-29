import torch.nn
import torch.nn as nn

from module import attention, encoder, common
from util import consts
from . import config
from ..model_meta import MetaType, model

import logging
logger = logging.getLogger(__name__)


@model("meta_din_ood", MetaType.ModelBuilder)
class DeepInterestNetwork(nn.Module):
    def __init__(self, model_conf):
        super(DeepInterestNetwork, self).__init__()

        assert isinstance(model_conf, config.ModelConfig)
        self._id_encoder = encoder.IDEncoder(
            model_conf.id_vocab,
            model_conf.id_dimension,
        )
        self._target_trans = common.StackedDense(
            model_conf.id_dimension, [model_conf.id_dimension], [torch.nn.Tanh]
        )
        self._seq_trans = common.StackedDense(
            model_conf.id_dimension, [model_conf.id_dimension], [torch.nn.Tanh]
        )

        self._target_attention = attention.TargetAttention(
            key_dimension=model_conf.id_dimension,
            value_dimension=model_conf.id_dimension,
        )

        self._meta_classifier_param_list = common.HyperNetwork_FC(
            # self._meta_classifier_param_list = common.HyperNetwork_FC_ood(
            # model_conf.id_dimension * 2,
            model_conf.id_dimension,
            model_conf.classifier + [1],
            ([torch.nn.Tanh] * len(model_conf.classifier)) + [None],
            batch=True,
            trigger_sequence_len=30,
            model_conf=model_conf,
            expand=True
        )
        
        self.stage2_mlp_trans = common.StackedDense(
            model_conf.id_dimension, [model_conf.id_dimension], [torch.nn.Tanh]
        )
        self.stage2_ood_classifier = common.StackedDense(
            model_conf.id_dimension * 2,
            model_conf.classifier + [1],
            ([torch.nn.Tanh] * len(model_conf.classifier)) + [None]
        )

        # self._classifier = common.StackedDense(
        #     model_conf.id_dimension * 2,
        #     model_conf.classifier + [1],
        #     ([torch.nn.Tanh] * len(model_conf.classifier)) + [None]
        # )

        
    def __setitem__(self, k, v):
        self.k = v

    def forward(self, features, pred=False, train_ood_threshold=0.99, stage=0, fig1=False):
        # Encode target item
        # B * D
        # print("-" * 50)
        # print(features)
        trigger_embed = self._id_encoder(features[consts.FIELD_TRIGGER_SEQUENCE])
        trigger_embed = self._seq_trans(trigger_embed)

        # request_embed = self._id_encoder(features[consts.FIELD_REQ_SEQUENCE])
        # request_embed = self._seq_trans(request_embed)

        target_embed = self._id_encoder(features[consts.FIELD_TARGET_ID])
        target_embed = self._target_trans(target_embed)

        # Encode user historical behaviors
        with torch.no_grad():
            click_mask = torch.not_equal(features[consts.FIELD_CLK_SEQUENCE], 0).to(dtype=torch.int32)
            trigger_mask = torch.not_equal(features[consts.FIELD_TRIGGER_SEQUENCE], 0).to(dtype=torch.int32)
            # B
            seq_length = torch.maximum(torch.sum(click_mask, dim=1) - 1, torch.Tensor([0]).to(device=click_mask.device))
            seq_length = seq_length.to(torch.long)
            trigger_seq_length = torch.maximum(torch.sum(trigger_mask, dim=1) - 1,
                                               torch.Tensor([0]).to(device=trigger_mask.device))
            trigger_seq_length = trigger_seq_length.to(torch.long)

        # B * L * D
        hist_embed = self._id_encoder(features[consts.FIELD_CLK_SEQUENCE])
        hist_embed = self._seq_trans(hist_embed)

        # Target attention
        atten_aggregated_embed = self._target_attention(
            target_key=target_embed,
            item_keys=hist_embed,
            item_values=hist_embed,
            mask=click_mask
        )

        atten_aggregated_trigger_embed = self._target_attention(
            target_key=target_embed,
            item_keys=trigger_embed,
            item_values=trigger_embed,
            mask=trigger_mask
        )

        classifier_input = torch.cat([target_embed, atten_aggregated_embed], dim=1)
        if not pred:
            # if use_duet_net:
            if stage == 1:
                # user_embedding = self._meta_classifier_param_list(atten_aggregated_embed, trigger_embed,
                #                                                   atten_aggregated_embed.size()[0],
                #                                                   trigger_seq_length)

                # user_embedding = self._meta_classifier_param_list(atten_aggregated_embed, hist_embed,
                #                                                   atten_aggregated_embed.size()[0],
                #                                                   seq_length)
                #
                # output = torch.sum(user_embedding * target_embed, dim=1, keepdim=True)
                output = self._meta_classifier_param_list(classifier_input, hist_embed,
                                                                  classifier_input.size()[0],
                                                                  seq_length)
                return output
            elif stage == 2:
                user_embedding = torch.mean(self.stage2_mlp_trans(hist_embed), dim=1)
                trigger_embedding = torch.mean(self.stage2_mlp_trans(trigger_embed), dim=1)
                output = self._meta_classifier_param_list(classifier_input, trigger_embed,
                                                          classifier_input.size()[0],
                                                          trigger_seq_length)
                # print(atten_aggregated_trigger_embed.size())
                # print(atten_aggregated_embed.size())
                # print(trigger_embedding.size())
                # print(user_embedding.size())
                # print(trigger_embed.size())
                # print(hist_embed.size())
                ood_pred = self.stage2_ood_classifier(torch.cat([trigger_embedding, user_embedding], dim=1))
                return output, ood_pred

        else:
            if stage == 1:
                # user_embedding = self._meta_classifier_param_list(atten_aggregated_embed, hist_embed,
                #                                                   atten_aggregated_embed.size()[0],
                #                                                   seq_length)
                #
                # output = torch.sum(user_embedding * target_embed, dim=1, keepdim=True)
                output = self._meta_classifier_param_list(classifier_input, hist_embed,
                                                                  classifier_input.size()[0],
                                                                  seq_length)
                return output

            elif stage == 2:
                user_embedding = torch.mean(self.stage2_mlp_trans(hist_embed), dim=1)
                trigger_embedding = torch.mean(self.stage2_mlp_trans(trigger_embed), dim=1)
                output = self._meta_classifier_param_list(classifier_input, trigger_embed,
                                                          classifier_input.size()[0],
                                                          trigger_seq_length)
                # if fig1:
                #     from copy import deepcopy
                #     output1 = deepcopy(output)
                
                ood_pred = self.stage2_ood_classifier(torch.cat([trigger_embedding, user_embedding], dim=1))

                # request_num = 0
                total_num = ood_pred.size()[0]

                request_index = torch.where(torch.sigmoid(ood_pred).detach().cpu() < train_ood_threshold, True, False).view(-1)

                request_num = torch.sum(torch.where(request_index, 1, 0))
                if request_num > 0:
                    # trigger_embed = hist_embed
                    output[request_index] = \
                        self._meta_classifier_param_list(classifier_input[request_index], hist_embed[request_index],
                                                         classifier_input[request_index].size()[0],
                                                         trigger_seq_length=seq_length[request_index]
                                                         )
                if fig1:
                    output1 = self._meta_classifier_param_list(classifier_input, trigger_embed,
                                                               classifier_input.size()[0],
                                                               trigger_seq_length)
                    output = self._meta_classifier_param_list(classifier_input, hist_embed,
                                                              classifier_input.size()[0],
                                                              seq_length)
                    return output, ood_pred, request_num, total_num, output1
                return output, ood_pred, request_num, total_num

        # if not pred:
        #     # return self._classifier(classifier_input)
        #     output, trigger_embed_ = self._meta_classifier_param_list(classifier_input, hist_embed,
        #                                      classifier_input.size()[0],
        #                                      trigger_seq_length=seq_length
        #                                      )
        # elif pred:
        #     output, trigger_embed_ = self._meta_classifier_param_list(classifier_input, trigger_embed,
        #                                                               classifier_input.size()[0],
        #                                                               trigger_seq_length=trigger_seq_length
        #                                                               )
        # # print("1 {} {}".format(target_embed.size(), atten_aggregated_embed.size()))
        # # print("1 {}".format(torch.cat([target_embed, atten_aggregated_embed], dim=1).size()))
        # # print("2 {} {}".format(trigger_embed.size(), hist_embed.size()))
        # # print("2 {} {}".format(trigger_embed_.size(), hist_embed.size()))
        # # ood_pred = self._ood_classifier(torch.cat([trigger_embed_, hist_embed], dim=1))
        #
        # # ood_pred = self._ood_classifier(torch.cat([trigger_embed_, atten_aggregated_embed], dim=1))
        # ood_pred = self._ood_classifier(torch.cat([atten_aggregated_trigger_embed, atten_aggregated_embed], dim=1))
        # if pred:
        #     # request_num = 0
        #     total_num = ood_pred.size()[0]
        #
        #     request_index = torch.where(torch.sigmoid(ood_pred).detach().cpu() < train_ood_threshold, True, False).view(-1)
        #     # print(request_index)
        #     # print(request_index.size())
        #     # print("-" * 50)
        #     # print(torch.sigmoid(ood_pred).detach().cpu())
        #     # print(request_index)
        #     # print(torch.argmax(torch.Tensor(ood_pred.detach().cpu()), dim=1))
        #     # print(request_index)
        #     request_num = torch.sum(torch.where(request_index, 1, 0))
        #     if request_num > 0:
        #         # trigger_embed = hist_embed
        #         output[request_index], _ = \
        #             self._meta_classifier_param_list(classifier_input[request_index], hist_embed[request_index],
        #                                              classifier_input[request_index].size()[0],
        #                                              trigger_seq_length=seq_length[request_index]
        #                                              )
        #
        #     return output, ood_pred, request_num, total_num
        # return output, ood_pred


class TestModule(nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.conv = nn.Conv2d(1, 20, 5)
        self.params = nn.ModuleDict({"conv": self.conv})
        self.params2 = nn.ModuleDict({"conv": self.conv})

    def forward(self, x):
        return self.conv(x)


if __name__ == '__main__':
    m = TestModule()
    for idx,p in enumerate(m.parameters()):
        print(idx, ":", p)
