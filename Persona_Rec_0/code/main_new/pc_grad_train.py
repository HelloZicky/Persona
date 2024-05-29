# coding=utf-8
import os
import time
import json
import logging
import math
import argparse
import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.multiprocessing as mp
from torch import nn

import model
from module.gradient_aggregation.PCGrad_repo import pcgrad
# from module.gradient_aggregation.PCGrad_repo import pcgrad_merge as pcgrad

from util.timer import Timer
# from util import oss_io
from util import args_processing as ap
from util import consts
from util import path
from util import env
# from loader import sequence_dataloader
from loader import multi_metric_meta_ood_sequence_dataloader as sequence_dataloader
from collections import defaultdict

from util import new_metrics
import numpy as np
from util import utils
utils.setup_seed(0)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

from copy import deepcopy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", type=str, help="Kernels configuration for CNN")
    parser.add_argument("--bucket", type=str, default=None, help="Bucket name for external storage")
    parser.add_argument("--dataset", type=str, default="alipay", help="Bucket name for external storage")

    parser.add_argument("--max_steps", type=int, help="Number of iterations before stopping")
    parser.add_argument("--snapshot", type=int, help="Number of iterations to dump model")
    parser.add_argument("--checkpoint_dir", type=str, help="Path of the checkpoint path")
    parser.add_argument("--learning_rate", type=str, default=0.001)
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")

    parser.add_argument("--max_epoch", type=int, default=10, help="Max epoch")
    parser.add_argument("--num_loading_workers", type=int, default=4, help="Number of threads for loading")
    parser.add_argument("--model", type=str, help="model type")
    parser.add_argument("--init_checkpoint", type=str, default="", help="Path of the checkpoint path")
    parser.add_argument("--init_step", type=int, default=0, help="Path of the checkpoint path")
    parser.add_argument("--group_num", type=int, default=5, help="Path of the checkpoint path")

    parser.add_argument("--max_gradient_norm", type=float, default=0.)

    # If both of the two options are set, `model_config` is preferred
    parser.add_argument("--arch_config_path", type=str, default=None, help="Path of model configs")
    parser.add_argument("--arch_config", type=str, default=None, help="base64-encoded model configs")

    return parser.parse_known_args()[0]


def predict_best(predict_dataloader_list, model_obj_list, device, args, train_epoch, train_step, writer=None):
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
    for model_obj in model_obj_list:
        model_obj.eval()
        model_obj.to(device)

    timer = Timer()
    log_every = 200
    WRITING_BATCH_SIZE = 512
    pred_list = []
    y_list = []
    buffer = []
    user_id_list = []
    # for step, batch_data_list in enumerate(zip(*train_dataset_list), 1):
    #     for group_index, batch_data in enumerate(batch_data_list):
    # for index, predict_dataloader in enumerate(predict_dataloader_list):
    #     for step, batch_data in enumerate(predict_dataloader, 1):
    for step, batch_data_list in enumerate(zip(*predict_dataloader_list), 1):
        for group_index, batch_data in enumerate(batch_data_list):
            logits = model_obj_list[group_index]({
                key: value.to(device)
                for key, value in batch_data.items()
                if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL, consts.FIELD_TRIGGER_SEQUENCE}
            })

            prob = torch.sigmoid(logits).detach().cpu().numpy()
            y = batch_data[consts.FIELD_LABEL].view(-1, 1)
            # print(logits)
            # print(prob)
            # print(y)
            # fpr, tpr, thresholds = new_metrics.roc_curve(np.array(y), prob, pos_label=1)
            # overall_auc = float(new_metrics.overall_auc(fpr, tpr))
            overall_auc, _, _, _ = new_metrics.calculate_overall_auc(prob, y)
            # ndcg = 0
            user_id_list.extend(np.array(batch_data[consts.FIELD_USER_ID].view(-1, 1)))
            pred_list.extend(prob)
            y_list.extend(np.array(y))

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
            if step % log_every == 0:
                logger.info(
                    "train_epoch={}, step={}, overall_auc={:5f}, speed={:2f} steps/s".format(
                        train_epoch, step, overall_auc, log_every / timer.tick(False)
                    )
                )
    # print(np.array(pred_list).shape, np.array(y_list).shape)
    overall_auc, _, _, _ = new_metrics.calculate_overall_auc(np.array(pred_list), np.array(y_list))
    user_auc = new_metrics.calculate_user_auc(buffer)
    overall_logloss = new_metrics.calculate_overall_logloss(np.array(pred_list), np.array(y_list))
    # user_ndcg, user_hr = new_metrics.calculate_user_ndcg_hr(10, np.array(pred_list), np.array(y_list))
    user_ndcg5, user_hr5 = new_metrics.calculate_user_ndcg_hr(5, buffer)
    user_ndcg10, user_hr10 = new_metrics.calculate_user_ndcg_hr(10, buffer)
    user_ndcg20, user_hr20 = new_metrics.calculate_user_ndcg_hr(20, buffer)
    # ndcg = 0

    print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
          "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
          format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
                 user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20))
    with open(os.path.join(args.checkpoint_dir, "log_ood_{}.txt".format(args.group_num)), "a") as writer:
        print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
              "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
              format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
                     user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20), file=writer)

    return overall_auc, user_auc, overall_logloss, user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20


def predict(predict_dataloader_list, model_obj_list, device, args, train_epoch, train_step, writer=None):
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
    for model_obj in model_obj_list:
        model_obj.eval()
        model_obj.to(device)

    timer = Timer()
    log_every = 200
    WRITING_BATCH_SIZE = 512

    user_id_list = []

    overall_auc_list, user_auc_list, overall_logloss_list, user_ndcg5_list, user_hr5_list, \
    user_ndcg10_list, user_hr10_list, user_ndcg20_list, user_hr20_list = [], [], [], [], [], [], [], [], []
    # for step, batch_data_list in enumerate(zip(*train_dataset_list), 1):
    #     for group_index, batch_data in enumerate(batch_data_list):
    # for index, predict_dataloader in enumerate(predict_dataloader_list):
    #     for step, batch_data in enumerate(predict_dataloader, 1):
    for step, batch_data_list in enumerate(zip(*predict_dataloader_list), 1):
        for group_index, batch_data in enumerate(batch_data_list):
            pred_list = []
            y_list = []
            buffer = []
            logits = model_obj_list[group_index]({
                key: value.to(device)
                for key, value in batch_data.items()
                if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL, consts.FIELD_TRIGGER_SEQUENCE}
            })

            prob = torch.sigmoid(logits).detach().cpu().numpy()
            y = batch_data[consts.FIELD_LABEL].view(-1, 1)
            # print(logits)
            # print(prob)
            # print(y)
            # fpr, tpr, thresholds = new_metrics.roc_curve(np.array(y), prob, pos_label=1)
            # overall_auc = float(new_metrics.overall_auc(fpr, tpr))
            overall_auc, _, _, _ = new_metrics.calculate_overall_auc(prob, y)
            # ndcg = 0
            user_id_list.extend(np.array(batch_data[consts.FIELD_USER_ID].view(-1, 1)))
            pred_list.extend(prob)
            y_list.extend(np.array(y))

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
            if step % log_every == 0:
                logger.info(
                    "train_epoch={}, step={}, overall_auc={:5f}, speed={:2f} steps/s".format(
                        train_epoch, step, overall_auc, log_every / timer.tick(False)
                    )
                )
            # print(np.array(pred_list).shape, np.array(y_list).shape)
            overall_auc, _, _, _ = new_metrics.calculate_overall_auc(np.array(pred_list), np.array(y_list))
            user_auc = new_metrics.calculate_user_auc(buffer)
            overall_logloss = new_metrics.calculate_overall_logloss(np.array(pred_list), np.array(y_list))
            # user_ndcg, user_hr = new_metrics.calculate_user_ndcg_hr(10, np.array(pred_list), np.array(y_list))
            user_ndcg5, user_hr5 = new_metrics.calculate_user_ndcg_hr(5, buffer)
            user_ndcg10, user_hr10 = new_metrics.calculate_user_ndcg_hr(10, buffer)
            user_ndcg20, user_hr20 = new_metrics.calculate_user_ndcg_hr(20, buffer)
            # ndcg = 0
            overall_auc_list.append(overall_auc)
            user_auc_list.append(user_auc)
            overall_logloss_list.append(overall_logloss)
            user_ndcg5_list.append(user_ndcg5)
            user_hr5_list.append(user_hr5)
            user_ndcg10_list.append(user_ndcg10)
            user_hr10_list.append(user_hr10)
            user_ndcg20_list.append(user_ndcg20)
            user_hr20_list.append(user_hr20)

    # print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
    #       "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
    #       format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
    #              user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20))
    # with open(os.path.join(args.checkpoint_dir, "log_ood_{}.txt".format(args.group_num)), "a") as writer:
    #     print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
    #           "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
    #           format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
    #                  user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20), file=writer)

    # return overall_auc, user_auc, overall_logloss, user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20
    return overall_auc_list, user_auc_list, overall_logloss_list, user_ndcg5_list, user_hr5_list, \
           user_ndcg10_list, user_hr10_list, user_ndcg20_list, user_hr20_list


# def train(train_dataset_list, model_obj, device, args, bucket):
def train(train_dataloader_list, predict_dataloader_list, model_obj_list, device, args):
    criterion = nn.BCEWithLogitsLoss()
    optimizer_pcgrad_list = []
    optimizer_list = []
    for model_obj in model_obj_list:
        optimizer = torch.optim.Adam(
            model_obj.parameters(),
            lr=float(args.learning_rate)
        )
        optimizer_list.append(optimizer)
        # optimizer_pcgrad = pcgrad.PCGrad(optimizer)
        # optimizer_pcgrad_list.append(optimizer_pcgrad)
        # optimizer_pcgrad = None
        # gradient_dict = {}
        # gradient_size_dict = {}
        # gradient_1d_size_dict = {}
        # for name, parms in model_obj.named_parameters():
        #     gradient_dict[name] = torch.zeros((args.group_num, parms.view(-1).size()[0])).to(device)
        #     gradient_size_dict[name] = parms.size()
        #     gradient_1d_size_dict[name] = parms.view(-1).size()
        model_obj.train()
        model_obj.to(device)
    optimizer_pcgrad = pcgrad.PCGrad(optimizer_list)
    # print(model_obj)

    logger.info("Start training...")
    timer = Timer()
    log_every = 200
    buffer = []
    best_auc = 0
    max_step = 0
    best_auc_ckpt_path_list = []
    best_model_list = [None for i in range(args.group_num)]
    best_auc_list = [0 for i in range(args.group_num)]
    for group_index in range(args.group_num):
        best_auc_ckpt_path = os.path.join(args.checkpoint_dir, "best_auc_{}_{}".format(args.group_num, group_index) + ".pkl")
        best_auc_ckpt_path_list.append(best_auc_ckpt_path)
    # print("=" * 50)
    # print("train_dataloader_list ", train_dataloader_list)
    # print("predict_dataloader_list ", predict_dataloader_list)
    # for step, batch_data_list in enumerate(zip(*train_dataloader_list), 1):
    #     print(step)
    for epoch in range(1, args.max_epoch + 1):
        for step, batch_data_list in enumerate(zip(*train_dataloader_list), 1):
            loss_per_group = 0
            losses = []
            for group_index, batch_data in enumerate(batch_data_list):
                # print("=" * 50)
                # print("batch_data_list", batch_data_list)
                # logits = model_obj({
                logits = model_obj_list[group_index]({
                    key: value.to(device)
                    for key, value in batch_data.items()
                    if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
                })
                loss = criterion(logits, batch_data[consts.FIELD_LABEL].view(-1, 1).to(device))
                losses.append(loss)
                loss_per_group += loss * batch_data[consts.FIELD_LABEL].size()[0]

                pred, y = torch.sigmoid(logits), batch_data[consts.FIELD_LABEL].view(-1, 1)
                # print("=" * 50)
                # print(pred)
                # print("-" * 50)
                # print(y)
                # print()
                overall_auc, _, _, _ = new_metrics.calculate_overall_auc(np.array(pred.detach().cpu()), np.array(y))

                if step % log_every == 0:
                    logger.info(
                        "epoch={}, step={}, loss={:5f}, overall_auc={:5f}, speed={:2f} steps/s".format(
                            epoch, step, float(loss.item()), overall_auc, log_every / timer.tick(False)
                        )
                    )
                max_step = step
                # buffer.extend(
                #     # [str(user_id), float(score), float(label)]
                #     [int(user_id), float(score), float(label)]
                #     for user_id, score, label
                #     in zip(
                #         batch_data[consts.FIELD_USER_ID],
                #         prob,
                #         batch_data[consts.FIELD_LABEL]
                #     )
                # )

            # optimizer = PCGrad(tf.train.AdamOptimizer())  # wrap your favorite optimizer
            # losses =  # a list of per-task losses
            assert len(losses) == args.group_num
            optimizer_pcgrad.pc_backward(losses)
            optimizer_pcgrad.step()
            # optimizer_pcgrad_list.pc_backward(losses)
            # for optimizer_pcgrad in optimizer_pcgrad_list:
            #     optimizer_pcgrad.step()

            if step % log_every == 0:
                logger.info(
                    "step={}, loss={:5f}, speed={:2f} steps/s".format(
                        step, float(loss_per_group.item() / args.batch_size), log_every / timer.tick(False)
                    )
                )

        pred_overall_auc_list, pred_user_auc_list, pred_overall_logloss_list, pred_user_ndcg5_list, pred_user_hr5_list, \
        pred_user_ndcg10_list, pred_user_hr10_list, pred_user_ndcg20_list, pred_user_hr20_list = predict(
            # predict_dataset=pred_dataloader,
            predict_dataloader_list=predict_dataloader_list,
            model_obj_list=model_obj_list,
            device=device,
            args=args,
            # bucket=bucket,
            train_epoch=epoch,
            train_step=epoch * max_step,
            # writer=writer
        )
        # ckpt_path = os.path.join(args.checkpoint_dir, consts.FILE_CHECKPOINT_PREFIX + str(step) + ".pkl")

        # ckpt_path = os.path.join(args.checkpoint_dir, consts.FILE_CHECKPOINT_PREFIX + "epoch_" + str(epoch) + ".pkl")
        # torch.save(model_obj, ckpt_path)
        logger.info("dump checkpoint for epoch {}".format(epoch))
        for group_index, model_obj in enumerate(model_obj_list):
            model_obj.train()
            if pred_overall_auc_list[group_index] > best_auc_list[group_index]:
                best_auc_list[group_index] = pred_overall_auc_list[group_index]
                best_model_list[group_index] = deepcopy(model_obj)
                torch.save(model_obj, best_auc_ckpt_path_list[group_index])

        pred_overall_auc, pred_user_auc, pred_overall_logloss, pred_user_ndcg5, pred_user_hr5, \
        pred_user_ndcg10, pred_user_hr10, pred_user_ndcg20, pred_user_hr20 = predict_best(
            # predict_dataset=pred_dataloader,
            predict_dataloader_list=predict_dataloader_list,
            model_obj_list=best_model_list,
            device=device,
            args=args,
            # bucket=bucket,
            train_epoch=epoch,
            train_step=epoch * max_step,
            # writer=writer
        )

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
    # with open(os.path.join(args.checkpoint_dir, "log_ood.txt"), "a") as writer:
    #     print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
    #           "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
    #           format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
    #                  user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20), file=writer)
    #
    # return overall_auc, user_auc, overall_logloss, user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20


def main_worker(_):
    args = parse_args()
    ap.print_arguments(args)

    args.checkpoint_dir = os.path.join(args.checkpoint_dir, "2_prototype_multi_long_tail")
    # bucket = oss_io.open_bucket(args.bucket)
    #
    # # Check if the specified path has an existed model
    # if bucket.object_exists(args.checkpoint_dir):
    #     raise ValueError("Model %s has already existed, please delete them and retry" % args.checkpoint_dir)
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    model_meta = model.get_model_meta(args.model)  # type: model.ModelMeta

    # Load model configuration
    # model_conf, raw_model_conf = ap.parse_arch_config_from_args(model_meta, args, bucket)  # type: dict
    model_conf, raw_model_conf = ap.parse_arch_config_from_args(model_meta, args)  # type: dict

    # Dump arguments and model architecture configuration to OSS
    # ap.dump_train_arguments(args.checkpoint_dir, args, bucket)
    # ap.dump_model_config(args.checkpoint_dir, raw_model_conf, bucket)

    # Construct model
    model_obj_list = []
    for i in range(args.group_num):
        model_obj = model_meta.model_builder(model_conf=model_conf)  # type: torch.nn.module
        model_obj_list.append(model_obj)
        model_obj = None
    device = env.get_device()

    # Setup up data loader
    train_dataset_list = []
    train_dataloader_list = []
    train_dataset_length_list = []

    predict_dataset_list = []
    predict_dataloader_list = []
    # dataset_length_list = {0: 1826232, 1: 2452783, 2: 3220428, 3: 4463935, 4: 9722539}
    # overall_dataset_length = 21688481
    # dataset_length_dict = {}
    dataset_length_list = []
    # train_files, test_files = args.dataset.split(';')
    train_files, test_files = [], []
    # for i in range(1, args.group_num + 1):
    for i in range(args.group_num):
        train_files.append(os.path.join(args.dataset, "group_{}".format(args.group_num), "train", "train_{}.txt".format(i)))
        test_files.append(os.path.join(args.dataset, "group_{}".format(args.group_num), "test", "test_{}.txt".format(i)))
    for group_index in range(args.group_num):
        # train_dataset = sequence_dataloader.SequenceDataLoader(
        train_dataset = sequence_dataloader.MetaSequenceDataLoader(
            # table_name=args.tables.split(',')[group_index],
            # table_name=train_files.split(',')[group_index],
            table_name=train_files[group_index],
            slice_id=0,
            slice_count=args.num_loading_workers,
            is_train=True,
        )
        # dataset_length_dict[group_index] = len(train_dataset)
        train_dataset_length_list.append(len(train_dataset))
        train_dataset_list.append(train_dataset)

        # predict_dataset = sequence_dataloader.SequenceDataLoader(
        predict_dataset = sequence_dataloader.MetaSequenceDataLoader(
            # table_name=args.tables.split(',')[group_index],
            # table_name=test_files.split(',')[group_index],
            table_name=test_files[group_index],
            slice_id=0,
            slice_count=args.num_loading_workers,
            is_train=False,
        )
        # dataset_length_dict[group_index] = len(train_dataset)
        # predict_dataset_length_list.append(len(predict_dataset))
        predict_dataset_list.append(predict_dataset)

    train_overall_dataset_length = sum(train_dataset_length_list)
    for group_index, (dataset_length, train_dataset, predict_dataset) in enumerate(zip(train_dataset_length_list, train_dataset_list, predict_dataset_list)):
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size * train_dataset_length_list[group_index] // train_overall_dataset_length,
            num_workers=args.num_loading_workers,
            pin_memory=True,
            collate_fn=train_dataset.batchify,
            drop_last=False
        )
        train_dataloader_list.append(train_dataloader)
        predict_dataloader = torch.utils.data.DataLoader(
            predict_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_loading_workers,
            pin_memory=True,
            collate_fn=predict_dataset.batchify,
            drop_last=False
        )
        predict_dataloader_list.append(predict_dataloader)
    # Setup training
    train(
        train_dataloader_list=train_dataloader_list,
        predict_dataloader_list=predict_dataloader_list,
        model_obj_list=model_obj_list,
        device=device,
        args=args,
        # bucket=bucket,
    )


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    mp.spawn(main_worker, nprocs=1)

