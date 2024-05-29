# coding=utf-8
import os
import time
import json
import logging
import math
import argparse
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.multiprocessing as mp
from torch import nn

import model
from util.timer import Timer
# from util import oss_io
from util import args_processing as ap
from util import consts
from util import path
from util import env
# from util import metrics
from util import new_metrics
# from loader import sequence_dataloader
# from loader import new_meta_sequence_dataloader2 as meta_sequence_dataloader
from loader import multi_metric_meta_ood_sequence_dataloader as meta_sequence_dataloader
import numpy as np
from thop import profile

from util import utils
utils.setup_seed(0)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=str, help="Kernels configuration for CNN")
    parser.add_argument("--bucket", type=str, default=None, help="Bucket name for external storage")
    parser.add_argument("--dataset", type=str, default="alipay", help="Bucket name for external storage")
    parser.add_argument("--grad_norm", type=float, default=0.5, help="Bucket name for external storage")

    parser.add_argument("--max_steps", type=int, help="Number of iterations before stopping")
    parser.add_argument("--snapshot", type=int, help="Number of iterations to dump model")
    parser.add_argument("--checkpoint_dir", type=str, help="Path of the checkpoint path")
    # parser.add_argument("--pretrain_model_path", type=str, default="", help="Path of the pretrain model path")
    parser.add_argument("--pretrain_base_model_path", type=str, default="", help="Path of the pretrain model path")
    parser.add_argument("--pretrain_meta_model_path", type=str, default="", help="Path of the pretrain model path")
    parser.add_argument('--pretrain', '-p', action='store_true', help='load pretrained model')
    parser.add_argument("--learning_rate", type=str, default=0.001)
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--group_num", type=int, default=5, help="Group Number")

    # parser.add_argument("--max_epoch", type=int, default=10, help="Max epoch")
    parser.add_argument("--max_epoch", type=int,  default=20, help="Max epoch")
    # parser.add_argument("--max_epoch", type=int,  default=5, help="Max epoch")
    parser.add_argument("--num_loading_workers", type=int, default=4, help="Number of threads for loading")
    parser.add_argument("--model", type=str, help="model type")
    parser.add_argument("--base_model", type=str, help="model type")
    parser.add_argument("--init_checkpoint", type=str, default="", help="Path of the checkpoint path")
    parser.add_argument("--init_step", type=int, default=0, help="Path of the checkpoint path")

    parser.add_argument("--max_gradient_norm", type=float, default=0.)

    # If both of the two options are set, `model_config` is preferred
    parser.add_argument("--arch_config_path", type=str, default=None, help="Path of model configs")
    parser.add_argument("--arch_config", type=str, default=None, help="base64-encoded model configs")

    return parser.parse_known_args()[0]


def freeze(model_obj):
    for name, parms in model_obj.named_parameters():
        # if name in model_load_name_set:
        # if name.split(".")[0] == "_classifier" or name.split(".")[0] == "_mlp_trans":
        #     # print("-" * 50)
        #     # print(name)
        #     # print(parms)
        #     parms.requires_grad = True
        # else:
        #     parms.requires_grad = False
        # if name.split(".")[0] != "_classifier" and name.split(".")[0] != "_mlp_trans":
        #     parms.requires_grad = False
        # else:
        #     print("require grad: ", name)
        if name.split(".")[0] == "_meta_classifier_param_list":
            parms.requires_grad = True
        else:
            parms.requires_grad = False
    return model_obj


def load_model(args, model_obj_meta, model_obj_base):
    # ckpt_path = os.path.join(args.checkpoint_dir, "base", "best_auc.pkl")
    base_ckpt_path = os.path.join(args.pretrain_base_model_path)
    meta_ckpt_path = os.path.join(args.pretrain_meta_model_path)
    # output_file = os.path.join(args.checkpoint_dir, "test.txt")
    # feature_file = os.path.join(args.checkpoint_dir, "feature.pt")

    base_model_load = torch.load(base_ckpt_path)
    base_model_load_name_set = set()
    for name, parms in base_model_load.named_parameters():
        base_model_load_name_set.add(name)
    print(base_model_load_name_set)

    meta_model_load = torch.load(meta_ckpt_path)
    meta_model_load_name_set = set()
    for name, parms in meta_model_load.named_parameters():
        meta_model_load_name_set.add(name)
    print(meta_model_load_name_set)

    # model_load_dict = model_load.state_dict()
    # model_obj_dict = model_obj.state_dict()
    # model_obj_dict.update(model_load_dict)
    # model_obj.load_state_dict(model_obj_dict)
    # # model_obj = freeze(model_obj)

    # return model_obj, model_load_name_set
    return base_model_load, base_model_load_name_set, meta_model_load, meta_model_load_name_set


# def predict(predict_dataset, model_obj, device, args, bucket, train_step, writer=None):
def predict(predict_dataset, model_obj, device, args, train_epoch, train_step, writer=None, pretrain_model_param_dict=None,
            y_list_overall=None, pred_list_overall=None, buffer_overall=None):
    # Load pretrained models

    # ckpt_path = path.join_oss_path(
    #     args.checkpoint_dir,
    #     consts.FILE_CHECKPOINT_PREFIX + str(args.step)
    # )
    # checkpoint = oss_io.load_checkpoint(
    #     bucket=bucket,
    #     model_path=ckpt_path
    # )
    # model_obj.load_state_dict(checkpoint[consts.STATE_MODEL])

    model_obj.eval()
    model_obj.to(device)

    timer = Timer()
    log_every = 200
    WRITING_BATCH_SIZE = 512
    pred_list = []
    y_list = []
    buffer = []
    user_id_list = []
    for step, batch_data in enumerate(predict_dataset, 1):
        logits = model_obj({
            key: value.to(device)
            for key, value in batch_data.items()
            if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
        }, pretrain_model=pretrain_model_param_dict, grad_norm=args.grad_norm)
        prob = torch.sigmoid(logits).detach().cpu().numpy()
        y = batch_data[consts.FIELD_LABEL].view(-1, 1)
        # overall_auc, _, _, _ = new_metrics.calculate_overall_auc(prob, y)
        # ndcg = 0
        user_id_list.extend(np.array(batch_data[consts.FIELD_USER_ID].view(-1, 1)))
        pred_list.extend(prob)
        y_list.extend(np.array(y))

        pred_list_overall.extend(prob)
        y_list_overall.extend(np.array(y))
        # buffer.extend(
        #     [str(user_id), float(score), float(label)]
        #     for user_id, score, label
        #     in zip(
        #         batch_data[consts.FIELD_USER_ID],
        #         batch_data[consts.FIELD_LABEL],
        #         prob
        #     )
        # )
        buffer.extend(
            # [str(user_id), float(score), float(label)]
            [int(user_id), float(score), float(label)]
            for user_id, score, label
            in zip(
                batch_data[consts.FIELD_USER_ID],
                prob,
                batch_data[consts.FIELD_LABEL]
            )
        )

        buffer_overall.extend(
            # [str(user_id), float(score), float(label)]
            [int(user_id), float(score), float(label)]
            for user_id, score, label
            in zip(
                batch_data[consts.FIELD_USER_ID],
                prob,
                batch_data[consts.FIELD_LABEL]
            )
        )
        # if step % log_every == 0:
        #     logger.info(
        #         "train_epoch={}, step={}, overall_auc={:5f}, speed={:2f} steps/s".format(
        #             train_epoch, step, overall_auc, log_every / timer.tick(False)
        #         )
        #     )
        if step % log_every == 0:
            logger.info(
                "train_epoch={}, step={}, speed={:2f} steps/s".format(
                    train_epoch, step, log_every / timer.tick(False)
                )
            )

        # if len(buffer) >= WRITING_BATCH_SIZE:
        #     writer.write(buffer, [0, 1, 2])
        #     buffer = []

    # # if len(buffer) >= 0:
    # #     writer.write(buffer, [0, 1, 2])
    #
    # overall_auc, _, _, _ = new_metrics.calculate_overall_auc(np.array(pred_list), np.array(y_list))
    # user_auc = new_metrics.calculate_user_auc(buffer)
    # overall_logloss = new_metrics.calculate_overall_logloss(np.array(pred_list), np.array(y_list))
    # # user_ndcg, user_hr = new_metrics.calculate_user_ndcg_hr(10, np.array(pred_list), np.array(y_list))
    # user_ndcg5, user_hr5 = new_metrics.calculate_user_ndcg_hr(5, buffer)
    # user_ndcg10, user_hr10 = new_metrics.calculate_user_ndcg_hr(10, buffer)
    # user_ndcg20, user_hr20 = new_metrics.calculate_user_ndcg_hr(20, buffer)
    # # ndcg = 0
    #
    # print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
    #       "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
    #       format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
    #              user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20))
    # # with open(os.path.join(args.checkpoint_dir, "log_ood.txt"), "a") as writer:
    # #     print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
    # #           "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
    # #           format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
    # #                  user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20), file=writer)
    #
    # return overall_auc, user_auc, overall_logloss, user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20, \
    #        pred_list, y_list, buffer
    return pred_list, y_list, buffer


# def train(train_dataset, model_obj, device, args, bucket, pred_dataloader):
# def train(train_dataset, model_obj, device, args, pred_dataloader, pretrain_model_param_dict):
def train(train_dataloader_list, predict_dataloader_list, base_model_obj, meta_model_obj, device, args, pretrain_model_param_dict=None,
          train_dataset_length_list=None, predict_dataset_length_list=None):

    # if args.pretrain:
    #     ckpt = torch.load(args.pretrain_model_path, map_location='cpu')
    #     ckpt.to(device)
    #     # weight = ckpt["net"]
    #     trunk_weight = {}
    #     # import ipdb
    #     # ipdb.set_trace()
    #     # param_dict = net.state_dict()
    #     pretrain_model_param_dict = ckpt.state_dict()
    #     # finetune_model_keys = set(list(net.state_dict().keys()))
    #     # pretrain_model_keys_set = weight.keys()
    #     # for index, name in enumerate(pretrain_model_keys_set):
    #     #     if name in finetune_model_keys:
    #     #         param_dict[name] = weight[name]
    #     # print(ckpt)
    # else:
    #     pretrain_model_param_dict = None

    pretrain_model_param_dict = base_model_obj.state_dict()

    criterion = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(
    #     meta_model_obj.parameters(),
    #     lr=float(args.learning_rate)
    # )
    # model_obj.train()
    # model_obj.to(device)
    from copy import deepcopy
    model_obj_backup = deepcopy(meta_model_obj)

    print(meta_model_obj)

    logger.info("Start training...")
    timer = Timer()
    log_every = 200
    # writer = open(os.path.join(args.checkpoint_dir, "log.txt"), "a")
    max_step = 0
    # best_auc = 0
    # best_user_auc = 0
    # best_logloss = 0
    # best_ndcg = 0
    # best_hr = 0
    # best_auc_ckpt_path = os.path.join(args.checkpoint_dir, "best_auc" + ".pkl")

    # best_user_auc_ckpt_path = os.path.join(args.checkpoint_dir, "best_user_auc" + ".pkl")
    # best_logloss_ckpt_path = os.path.join(args.checkpoint_dir, "best_logloss" + ".pkl")
    # best_ndcg_ckpt_path = os.path.join(args.checkpoint_dir, "best_ndcg" + ".pkl")
    # best_hr_ckpt_path = os.path.join(args.checkpoint_dir, "best_hr" + ".pkl")
    print("=" * 50)
    print("train_dataloader_list ", train_dataloader_list)
    print("predict_dataloader_list ", predict_dataloader_list)
    model_dict = {}
    best_auc_dict = {}
    optimizer_dict = {}
    best_auc_ckpt_path_dict = {}
    for group_index in range(args.group_num):
        model_dict[group_index] = deepcopy(model_obj_backup)
        best_auc_dict[group_index] = 0
        optimizer_dict[group_index] = torch.optim.Adam(
            model_dict[group_index].parameters(),
            lr=float(args.learning_rate)
        )
        best_auc_ckpt_path_dict[group_index] = os.path.join(args.checkpoint_dir, "best_auc_{}".format(group_index) + ".pkl")
    y_dict_overall = {}
    pred_dict_overall = {}
    buffer_dict_overall = {}

    best_auc = 0
    # for train_dataloader, pred_dataloader in zip(train_dataloader_list, predict_dataloader_list):
    for epoch in range(1, args.max_epoch + 1):
        y_list_overall = []
        pred_list_overall = []
        buffer_overall = []

        for group_index in range(args.group_num):
            print("-.-" * 50)
            print("group_index------>>>>", group_index)
            train_dataloader = train_dataloader_list[group_index]
            pred_dataloader = predict_dataloader_list[group_index]
            train_dataset_length = train_dataset_length_list[group_index]
            predict_dataset_length = predict_dataset_length_list[group_index]
            # model_obj = deepcopy(model_obj_backup)
            model_obj = model_dict[group_index]
            model_obj = freeze(model_obj)
            model_obj.train()
            model_obj.to(device)
            # best_auc = 0
            best_user_auc = 0
            best_logloss = 0
            best_ndcg = 0
            best_hr = 0
            # best_auc_ckpt_path = os.path.join(args.checkpoint_dir, "best_auc" + ".pkl")
            # best_auc_ckpt_path = os.path.join(args.checkpoint_dir, "best_auc_{}".format(group_index) + ".pkl")
            best_auc_ckpt_path = best_auc_ckpt_path_dict[group_index]
            # pred_overall_auc = 0

            # model_obj.train()
            for step, batch_data in enumerate(train_dataloader, 1):
                logits = model_obj({
                    key: value.to(device)
                    for key, value in batch_data.items()
                    if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
                }, pretrain_model=pretrain_model_param_dict, grad_norm=args.grad_norm)

                # if step == 1:
                #     # from thop import profile
                #     flops, params = profile(model_obj,
                #         {key: value.to(device)
                #          for key, value in batch_data.items()
                #          if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
                #          },)
                #     print("flops={}B, params={}M".format(flops / 1e9, params / 1e6))

                loss = criterion(logits, batch_data[consts.FIELD_LABEL].view(-1, 1).to(device))
                pred, y = torch.sigmoid(logits), batch_data[consts.FIELD_LABEL].view(-1, 1)
                # fpr, tpr, thresholds = new_metrics.roc_curve(np.array(y), np.array(pred.detach().cpu()), pos_label=1)
                # auc = float(new_metrics.auc(fpr, tpr))
                # print("np.array(pred.detach().cpu()) ", np.array(pred.detach().cpu()))
                # print("np.array(y) ", np.array(y))
                # print("np.array(pred.detach().cpu()).shape ", np.array(pred.detach().cpu()).shape)
                # print("np.array(y).shape ", np.array(y).shape)
                # auc = new_metrics.calculate_overall_auc(np.array(pred.detach().cpu()), np.array(y.int()))
                # auc, _, _, _ = new_metrics.calculate_overall_auc(np.array(pred.detach().cpu()), np.array(y))
                try:
                    # "ROC error"
                    auc, _, _, _ = new_metrics.calculate_overall_auc(np.array(pred.detach().cpu()), np.array(y))
                except ValueError:
                    auc = 0
                # optimizer.zero_grad()
                optimizer_dict[group_index].zero_grad()
                loss.backward()
                # optimizer.step()
                optimizer_dict[group_index].step()

                if step % log_every == 0:
                    logger.info(
                        "epoch={}, step={}, loss={:5f}, auc={:5f}, speed={:2f} steps/s".format(
                            epoch, step, float(loss.item()), auc, log_every / timer.tick(False)
                        )
                    )
                max_step = step
            if predict_dataset_length > 0:
                # pred_overall_auc, pred_user_auc, pred_overall_logloss, pred_user_ndcg5, pred_user_hr5, \
                # pred_user_ndcg10, pred_user_hr10, pred_user_ndcg20, pred_user_hr20, \
                pred_list, y_list, buffer = \
                predict(
                    predict_dataset=pred_dataloader,
                    model_obj=model_obj,
                    device=device,
                    args=args,
                    # bucket=bucket,
                    train_epoch=epoch,
                    train_step=epoch * max_step,
                    # writer=writer,
                    pretrain_model_param_dict=pretrain_model_param_dict,
                    y_list_overall=y_list_overall,
                    pred_list_overall=pred_list_overall,
                    buffer_overall=buffer_overall
                )
                y_list_overall.extend(y_list)
                pred_list_overall.extend(pred_list)
                buffer_overall.extend(buffer)
                # ckpt_path = os.path.join(args.checkpoint_dir, consts.FILE_CHECKPOINT_PREFIX + str(step) + ".pkl")

                # ckpt_path = os.path.join(args.checkpoint_dir, consts.FILE_CHECKPOINT_PREFIX + "epoch_" + str(epoch) + ".pkl")
                # torch.save(model_obj, ckpt_path)
                logger.info("dump checkpoint for epoch {}".format(epoch))
                # model_obj.train()
                # if pred_overall_auc > best_auc_dict[group_index]:
                #     model_dict[group_index] = model_obj
                #     # best_auc = pred_overall_auc
                #     best_auc_dict[group_index] = pred_overall_auc
                #     torch.save(model_obj, best_auc_ckpt_path)
                #     y_dict_overall[group_index] = y_list
                #     pred_dict_overall[group_index] = pred_list
                #     buffer_dict_overall[group_index] = buffer
                # torch.save(model_obj, best_auc_ckpt_path)

            # else:
            #     logger.info("dump checkpoint for epoch {}".format(epoch))
            #     model_dict[group_index] = model_obj
            #     torch.save(model_obj, best_auc_ckpt_path)

        print(len(pred_list_overall))
        print(len(y_list_overall))
        # ndcg = 0
        assert len(pred_list_overall) == len(y_list_overall)
        overall_auc, _, _, _ = new_metrics.calculate_overall_auc(np.array(pred_list_overall), np.array(y_list_overall))
        user_auc = new_metrics.calculate_user_auc(buffer_overall)
        overall_logloss = new_metrics.calculate_overall_logloss(np.array(pred_list_overall), np.array(y_list_overall))
        # user_ndcg, user_hr = new_metrics.calculate_user_ndcg_hr(10, np.array(pred_list), np.array(y_list))
        user_ndcg5, user_hr5 = new_metrics.calculate_user_ndcg_hr(5, buffer_overall)
        user_ndcg10, user_hr10 = new_metrics.calculate_user_ndcg_hr(10, buffer_overall)
        user_ndcg20, user_hr20 = new_metrics.calculate_user_ndcg_hr(20, buffer_overall)
        # ndcg = 0

        print("epoch={}, step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
              "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
              format(epoch, 1, overall_auc, user_auc, overall_logloss,
                     user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20))
        with open(os.path.join(args.checkpoint_dir, "log_ood.txt"), "a") as writer:
            print("epoch={}, step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
                  "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
                  format(epoch, 1, overall_auc, user_auc, overall_logloss,
                         user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20), file=writer)
        # pred_overall_auc = overall_auc
        # # if pred_overall_auc > best_auc_dict[group_index]:
        # if pred_overall_auc > best_auc:
        #     # model_dict[group_index] = model_obj
        #     # # best_auc = pred_overall_auc
        #     # best_auc_dict[group_index] = pred_overall_auc
        #     # torch.save(model_obj, best_auc_ckpt_path)
        #     # y_dict_overall[group_index] = y_list
        #     # pred_dict_overall[group_index] = pred_list
        #     # buffer_dict_overall[group_index] = buffer
        #     for group_index in range(args.group_num):
        #         torch.save(model_dict[group_index], best_auc_ckpt_path_dict[group_index])
        pred_user_auc = user_auc
        # if pred_overall_auc > best_auc_dict[group_index]:
        if pred_user_auc > best_auc:
            for group_index in range(args.group_num):
                torch.save(model_dict[group_index], best_auc_ckpt_path_dict[group_index])
            best_auc = pred_user_auc
        # # auc = metrics.calculate_auc(np.array(pred_list_overall), np.array(y_list_overall))
        # # ndcg = metrics.calculate_ndcg(10, np.array(pred_list), np.array(y_list))
        # print(len(pred_list_overall))
        # print(len(y_list_overall))
        # # ndcg = 0
        # assert len(pred_list_overall) == len(y_list_overall)
        # # print("train_epoch={}, train_step={}, auc={:5f}, ndcg={:5f}".format(1, 1, auc, ndcg))
        # # with open(os.path.join(args.checkpoint_dir, "log_overall.txt"), "a") as writer:
        # #     print("train_epoch={}, train_step={}, auc={:5f}, ndcg={:5f}".format(1, 1, auc, ndcg), file=writer)
        # pred_list_overall = []
        # y_list_overall = []
        # for pred_list, y_list in zip(pred_dict_overall.values(), y_dict_overall.values()):
        #     pred_list_overall.extend(pred_list)
        #     y_list_overall.extend(y_list)
        # # pred_list_overall = pred_dict_overall.values()
        # # y_list_overall = y_dict_overall.values()
        # overall_auc, _, _, _ = new_metrics.calculate_overall_auc(np.array(pred_list_overall),
        #                                                          np.array(y_list_overall))
        # user_auc = new_metrics.calculate_user_auc(buffer_overall)
        # overall_logloss = new_metrics.calculate_overall_logloss(np.array(pred_list_overall),
        #                                                         np.array(y_list_overall))
        # # user_ndcg, user_hr = new_metrics.calculate_user_ndcg_hr(10, np.array(pred_list), np.array(y_list))
        # user_ndcg5, user_hr5 = new_metrics.calculate_user_ndcg_hr(5, buffer_overall)
        # user_ndcg10, user_hr10 = new_metrics.calculate_user_ndcg_hr(10, buffer_overall)
        # user_ndcg20, user_hr20 = new_metrics.calculate_user_ndcg_hr(20, buffer_overall)
        # # ndcg = 0
        #
        # print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
        #       "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
        #       # format(1, 1, overall_auc, user_auc, overall_logloss,
        #       format(epoch, 1, overall_auc, user_auc, overall_logloss,
        #              user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20))
        # with open(os.path.join(args.checkpoint_dir, "log_overall.txt"), "a") as writer:
        #     print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
        #           "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
        #           # format(1, 1, overall_auc, user_auc, overall_logloss,
        #           format(epoch, 1, overall_auc, user_auc, overall_logloss,
        #                  user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20), file=writer)

        # if pred_user_auc > best_user_auc:
        #     best_user_auc = pred_user_auc
        #     torch.save(model_obj, best_user_auc_ckpt_path)
        #
        # if pred_overall_logloss > best_logloss:
        #     best_logloss = pred_overall_logloss
        #     torch.save(model_obj, best_logloss_ckpt_path)
        #
        # if pred_user_ndcg > best_ndcg:
        #     best_ndcg = pred_user_ndcg
        #     torch.save(model_obj, best_ndcg_ckpt_path)
        #
        # if pred_user_hr > best_hr:
        #     best_hr = pred_user_hr
        #     torch.save(model_obj, best_hr_ckpt_path)


def main_worker(_):
    args = parse_args()
    ap.print_arguments(args)

    # bucket = oss_io.open_bucket(args.bucket)

    # Check if the specified path has an existed model
    # if bucket.object_exists(args.checkpoint_dir):
    # args.checkpoint_dir = os.path.join(args.checkpoint_dir, "meta")
    # if os.path.exists(args.checkpoint_dir):
    #     raise ValueError("Model %s has already existed, please delete them and retry" % args.checkpoint_dir)
    # else:
    #     os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    model_meta = model.get_model_meta(args.model)  # type: model.ModelMeta
    model_base = model.get_model_meta(args.base_model)  # type: model.ModelMeta

    # Load model configuration
    # model_conf, raw_model_conf = ap.parse_arch_config_from_args(model_meta, args, bucket)  # type: dict
    model_conf_meta, raw_model_conf_meta = ap.parse_arch_config_from_args(model_meta, args)  # type: dict
    model_conf_base, raw_model_conf_base = ap.parse_arch_config_from_args(model_base, args)  # type: dict

    # Dump arguments and model architecture configuration to OSS
    # ap.dump_train_arguments(args.checkpoint_dir, args, bucket)
    # ap.dump_model_config(args.checkpoint_dir, raw_model_conf, bucket)

    # Construct model
    model_obj_meta = model_meta.model_builder(model_conf=model_conf_meta)  # type: torch.nn.module
    model_obj_base = model_base.model_builder(model_conf=model_conf_base)  # type: torch.nn.module
    # model_obj, model_load_name_set = load_model(args, model_obj)
    base_model_obj, base_model_load_name_set, meta_model_obj, meta_model_load_name_set = load_model(args, model_obj_meta, model_obj_base)

    print("=" * 100)
    for name, parms in base_model_obj.named_parameters():
        print(name)
    print("-" * 50)
    for name, parms in meta_model_obj.named_parameters():
        print(name)
    print("=" * 100)
    device = env.get_device()
    # worker_id, worker_count = env.get_cluster_info()
    worker_id = worker_count = 8
    # train_file, test_file = args.dataset.split(',')

    # important!!!!
    args.num_loading_workers = 1

    # train_dataset_list = []
    train_dataset_list = {}
    # train_dataloader_list = []
    train_dataloader_list = {}
    # train_dataset_length_list = []
    train_dataset_length_list = {}

    # predict_dataset_list = []
    predict_dataset_list = {}
    # predict_dataloader_list = []
    predict_dataloader_list = {}
    predict_dataset_length_list = {}

    train_files, test_files = [], []
    # for i in range(1, args.group_num + 1):

    # for i in range(args.group_num):
    #     train_files.append(
    #         # os.path.join(args.dataset, "group_{}".format(args.group_num), "train", "train_{}.txt".format(i)))
    #         os.path.join(args.dataset, "domain_label_grad_random_5", args.base_model, args.group_num, "train_{}.txt".format(i)))
    #     test_files.append(
    #         # os.path.join(args.dataset, "group_{}".format(args.group_num), "test", "test_{}.txt".format(i)))
    #         os.path.join(args.dataset, "domain_label_grad_random_5", args.base_model, args.group_num, "test_{}.txt".format(i)))

    for group_index in range(args.group_num):
        # train_file = os.path.join(args.dataset, "domain_label_grad", args.base_model, str(args.group_num), "train_{}.txt".format(group_index))
        # test_file = os.path.join(args.dataset, "domain_label_grad", args.base_model, str(args.group_num), "test_{}.txt".format(group_index))
        train_file = os.path.join(args.dataset, "domain_label_grad_gru_random_0_final", args.base_model, str(args.group_num), "train_{}.txt".format(group_index))
        test_file = os.path.join(args.dataset, "domain_label_grad_gru_random_0_final", args.base_model, str(args.group_num), "test_{}.txt".format(group_index))
        # train_dataset = sequence_dataloader.SequenceDataLoader(
        # train_dataset = sequence_dataloader.MetaSequenceDataLoader(
        train_dataset = meta_sequence_dataloader.MetaSequenceDataLoader(
            # table_name=args.tables.split(',')[group_index],
            # table_name=train_files.split(',')[group_index],
            # table_name=train_files[group_index],
            table_name=train_file,
            slice_id=0,
            slice_count=args.num_loading_workers,
            is_train=True,
        )
        # dataset_length_dict[group_index] = len(train_dataset)
        # train_dataset_length_list.append(len(train_dataset))
        train_dataset_length_list[group_index] = len(train_dataset)
        # train_dataset_list.append(train_dataset)
        train_dataset_list[group_index] = train_dataset

        # predict_dataset = sequence_dataloader.SequenceDataLoader(
        # predict_dataset = sequence_dataloader.MetaSequenceDataLoader(
        predict_dataset = meta_sequence_dataloader.MetaSequenceDataLoader(
            # table_name=args.tables.split(',')[group_index],
            # table_name=test_files.split(',')[group_index],
            # table_name=test_files[group_index],
            table_name=test_file,
            slice_id=0,
            slice_count=args.num_loading_workers,
            is_train=False,
        )
        # dataset_length_dict[group_index] = len(train_dataset)
        # predict_dataset_length_list.append(len(predict_dataset))
        # predict_dataset_list.append(predict_dataset)
        predict_dataset_length_list[group_index] = len(predict_dataset)
        predict_dataset_list[group_index] = predict_dataset

    # train_overall_dataset_length = sum(train_dataset_length_list)
    train_overall_dataset_length = sum(train_dataset_length_list.values())
    # for group_index, (dataset_length, train_dataset, predict_dataset) in enumerate(
    #         zip(train_dataset_length_list, train_dataset_list, predict_dataset_list)):
    for group_index in range(args.group_num):
        train_dataset = train_dataset_list[group_index]
        predict_dataset = predict_dataset_list[group_index]
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            # batch_size=args.batch_size * train_dataset_length_list[group_index] // train_overall_dataset_length,
            batch_size=args.batch_size,
            num_workers=args.num_loading_workers,
            pin_memory=True,
            collate_fn=train_dataset.batchify,
            drop_last=False
        )
        # train_dataloader_list.append(train_dataloader)
        train_dataloader_list[group_index] = train_dataloader
        predict_dataloader = torch.utils.data.DataLoader(
            predict_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_loading_workers,
            pin_memory=True,
            collate_fn=predict_dataset.batchify,
            drop_last=False
        )
        # predict_dataloader_list.append(predict_dataloader)
        predict_dataloader_list[group_index] = predict_dataloader

    # # Setup up data loader
    # train_dataloader = meta_sequence_dataloader.MetaSequenceDataLoader(
    #     table_name=train_file,
    #     slice_id=0,
    #     slice_count=args.num_loading_workers,
    #     is_train=True
    # )
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataloader,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_loading_workers,
    #     pin_memory=True,
    #     collate_fn=train_dataloader.batchify,
    #     drop_last=False,
    #     # shuffle=True
    # )
    #
    # # Setup up data loader
    # pred_dataloader = meta_sequence_dataloader.MetaSequenceDataLoader(
    #     table_name=test_file,
    #     slice_id=args.num_loading_workers * worker_id,
    #     slice_count=args.num_loading_workers * worker_count,
    #     is_train=False
    # )
    # pred_dataloader = torch.utils.data.DataLoader(
    #     pred_dataloader,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_loading_workers,
    #     pin_memory=True,
    #     collate_fn=pred_dataloader.batchify,
    #     drop_last=False
    # )

    print("=" * 50)
    # if args.pretrain:
    #     ckpt = torch.load(args.pretrain_model_path, map_location='cpu')
    #     ckpt.to(device)
    #     # weight = ckpt["net"]
    #     trunk_weight = {}
    #     # import ipdb
    #     # ipdb.set_trace()
    #     # param_dict = net.state_dict()
    #     pretrain_model_param_dict = ckpt.state_dict()
    #     # finetune_model_keys = set(list(net.state_dict().keys()))
    #     # pretrain_model_keys_set = weight.keys()
    #     # for index, name in enumerate(pretrain_model_keys_set):
    #     #     if name in finetune_model_keys:
    #     #         param_dict[name] = weight[name]
    #     # print(ckpt)
    # else:
    #     pretrain_model_param_dict = None

    # print("*" * 50)
    # print(pretrain_model_param_dict)

    # Setup training
    train(
        # train_dataset=train_dataloader,
        train_dataloader_list=train_dataloader_list,
        predict_dataloader_list=predict_dataloader_list,
        # model_obj=model_obj,
        base_model_obj=base_model_obj,
        meta_model_obj=meta_model_obj,
        device=device,
        args=args,
        train_dataset_length_list=train_dataset_length_list,
        predict_dataset_length_list=predict_dataset_length_list,
        # bucket=bucket,
        # pred_dataloader=pred_dataloader,
        # pretrain_model_param_dict=pretrain_model_param_dict
    )


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    mp.spawn(main_worker, nprocs=1)

