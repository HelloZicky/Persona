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
# from loader import new_meta_sequence_dataloader as meta_sequence_dataloader
# from loader import new_meta_sequence_dataloader2 as meta_sequence_dataloader
from loader import multi_metric_meta_ood_sequence_dataloader as meta_sequence_dataloader
import numpy as np
from thop import profile
from tqdm import tqdm
random_seed = 0
# utils.setup_seed(random_seed)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


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
    # parser.add_argument("--max_epoch", type=int, default=10, help="Max epoch")
    parser.add_argument("--max_epoch", type=int, default=10, help="Max epoch")
    parser.add_argument("--num_loading_workers", type=int, default=4, help="Number of threads for loading")
    parser.add_argument("--model", type=str, help="model type")
    parser.add_argument("--init_checkpoint", type=str, default="", help="Path of the checkpoint path")
    parser.add_argument("--init_step", type=int, default=0, help="Path of the checkpoint path")

    parser.add_argument("--max_gradient_norm", type=float, default=0.)

    # If both of the two options are set, `model_config` is preferred
    parser.add_argument("--arch_config_path", type=str, default=None, help="Path of model configs")
    parser.add_argument("--arch_config", type=str, default=None, help="base64-encoded model configs")

    parser.add_argument("--group_num", type=int, default=5, help="base64-encoded model configs")
    parser.add_argument("--dataset_folder", type=str, default="", help="base64-encoded model configs")
    parser.add_argument("--base_model", type=str, default="", help="")

    return parser.parse_known_args()[0]


# def predict(predict_dataset, model_obj, device, args, bucket, train_step, writer=None):
def predict(predict_dataset, model_obj, device, args, train_epoch, train_step, writer=None, epoch=0, train_ood_threshold=0.5, best_model=None):
    if best_model is not None:
        model_obj = best_model
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

    # ood_prob_list = []
    # for step, batch_data in tqdm(enumerate(predict_dataset, 1)):
    #     logits, ood_logits, request_num, total_num = model_obj({
    #         key: value.to(device)
    #         for key, value in batch_data.items()
    #         if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
    #         # }, pred=True, train_ood_threshold=train_ood_threshold)
    #     }, pred=True, train_ood_threshold=args.train_ood_threshold)
    #     prob = torch.sigmoid(logits).detach().cpu().numpy()
    #     y = batch_data[consts.FIELD_LABEL].view(-1, 1)
    #     ood_prob = torch.sigmoid(ood_logits).detach().cpu().numpy()
    #     ood_prob_list.extend(ood_prob)

    # train_ood_threshold_list = []
    # # args.uncertainty_threshold = torch.Tensor([sorted(ood_prob_list)[len(ood_prob_list) * 10 // 100]])
    # train_ood_threshold = torch.Tensor([sorted(ood_prob_list)[len(ood_prob_list) * 10 // 100]])
    # train_ood_threshold_list.append(train_ood_threshold)
    with open(os.path.join(args.checkpoint_dir, "test.txt"), "w+") as writer:
        pred_list = []
        pred_label_list = []
        y_list = []
        ood_pred_list = []
        ood_y_list = []
        
        buffer = []
        _request_num = 0
        _total_num = 0
        user_id_list = []
        for step, batch_data in tqdm(enumerate(predict_dataset, 1)):
            logits = model_obj({
                key: value.to(device)
                for key, value in batch_data.items()
                if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
                # }, pred=True, train_ood_threshold=train_ood_threshold)
            })
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            y = batch_data[consts.FIELD_LABEL].view(-1, 1)
        
            # ood_prob = torch.sigmoid(ood_logits).detach().cpu().numpy()
            # print("=" * 50)
            # print(prob)
            # ood_y = torch.where(torch.argmax(torch.Tensor(prob), dim=1).view(-1, 1) == y, 1.0, 0.0)
        
            # ood_y = torch.where(torch.Tensor(prob).view(-1, 1) >= 0.1, 1.0, 0.0)
        
            # 10.27
            # pred_label = torch.where(torch.Tensor(prob).view(-1, 1) >= 0.1, 1.0, 0.0)
            pred_label = torch.where(torch.Tensor(prob).view(-1, 1) >= 0.5, 1.0, 0.0)
            ood_y = torch.where(pred_label == y, 1.0, 0.0)
        
            # print(ood_y)
            # auc = new_metrics.calculate_overall_auc(prob, y)
            # ood_auc = new_metrics.calculate_overall_auc(ood_prob, np.array(ood_y))
        
            # ndcg = new_metrics.calculate_ndcg(10, prob, y)
        
            # overall_auc = new_metrics.calculate_overall_auc(prob, y)
            # ndcg = 0
            user_id_list.extend(np.array(batch_data[consts.FIELD_USER_ID].view(-1, 1)))
        
            ndcg = 0
        
            # _request_num += request_num
            # _total_num += total_num
        
            # pred_list.extend(prob)
            # y_list.extend(np.array(y))
            # ood_pred_list.extend(ood_prob)
            # ood_y_list.extend(np.array(ood_y))
        
            # import ipdb
            # ipdb.set_trace()
        
            pred_list.extend(prob.tolist())
            # pred_label_list.extend(pred_label.detach().cpu().numpy().tolist())
            y_list.extend(np.array(y).tolist())
            # ood_pred_list.extend(ood_prob.tolist())
            # ood_y_list.extend(np.array(ood_y).tolist())
        
            # if epoch >= 5:
            #     print("=" * 50)
            #     print(ood_pred_list)
            #     print("-" * 50)
            #     print(ood_y_list)
        
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
        
        overall_auc, fpr, tpr, thresholds = new_metrics.calculate_overall_auc(np.array(pred_list), np.array(y_list))
        # _threshold = thresholds[len(thresholds) // 2]
        # _threshold = 0.5
        # pred_label = torch.where(torch.Tensor(pred_list).view(-1, 1) >= _threshold, 1.0, 0.0)
        # ood_y_list = torch.where(pred_label == torch.Tensor(y_list), 1.0, 0.0)
        user_auc = new_metrics.calculate_user_auc(buffer)
        # ood_auc, ood_fpr, ood_tpr, ood_thresholds = new_metrics.calculate_overall_auc(np.array(ood_pred_list), np.array(ood_y_list))
        # ood_auc, ood_fpr, ood_tpr, ood_thresholds = new_metrics.calculate_overall_auc(np.array(ood_pred_list), np.array(ood_y_list))
        # _ood_threshold = ood_thresholds[len(ood_thresholds) // 2]
        # _ood_threshold = 0.5
        overall_logloss = new_metrics.calculate_overall_logloss(np.array(pred_list), np.array(y_list))
        # user_ndcg, user_hr = new_metrics.calculate_user_ndcg_hr(10, np.array(pred_list), np.array(y_list))
        user_ndcg5, user_hr5 = new_metrics.calculate_user_ndcg_hr(5, buffer)
        user_ndcg10, user_hr10 = new_metrics.calculate_user_ndcg_hr(10, buffer)
        user_ndcg20, user_hr20 = new_metrics.calculate_user_ndcg_hr(20, buffer)
        # ndcg = 0
        
        # output_file =
        print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
              "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
              format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
                     user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20))

        print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
              "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
              format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
                     user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20), file=writer)
        
        # return overall_auc, user_auc, overall_logloss, user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20, ood_auc, _threshold, _ood_threshold


def predict_per_user(predict_dataset, model_obj, device, args, train_epoch, train_step, writer=None, epoch=0,
            train_ood_threshold=0.5, best_model=None):
    if best_model is not None:
        model_obj = best_model

    model_obj.eval()
    model_obj.to(device)

    timer = Timer()
    log_every = 200
    WRITING_BATCH_SIZE = 512

    with open(os.path.join(args.checkpoint_dir, "test_per_user.txt"), "a+") as writer:
        pred_list = []
        pred_label_list = []
        y_list = []
        ood_pred_list = []
        ood_y_list = []

        buffer = []
        _request_num = 0
        _total_num = 0
        user_id_list = []
        for step, batch_data in tqdm(enumerate(predict_dataset, 1)):
            logits = model_obj({
                key: value.to(device)
                for key, value in batch_data.items()
                if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
                # }, pred=True, train_ood_threshold=train_ood_threshold)
            })
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            y = batch_data[consts.FIELD_LABEL].view(-1, 1)

            user_id_list.extend(np.array(batch_data[consts.FIELD_USER_ID].view(-1, 1)))

            pred_list.extend(prob.tolist())
            # pred_label_list.extend(pred_label.detach().cpu().numpy().tolist())
            y_list.extend(np.array(y).tolist())

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

        # overall_auc, fpr, tpr, thresholds = new_metrics.calculate_overall_auc(np.array(pred_list), np.array(y_list))
        # overall_auc, fpr, tpr, thresholds = new_metrics.calculate_overall_auc(np.array(pred_list), np.array(y_list))
        overall_auc = 0

        user_auc = new_metrics.calculate_user_auc(buffer)

        overall_logloss = new_metrics.calculate_overall_logloss(np.array(pred_list), np.array(y_list))
        # user_ndcg, user_hr = new_metrics.calculate_user_ndcg_hr(10, np.array(pred_list), np.array(y_list))
        user_ndcg5, user_hr5 = new_metrics.calculate_user_ndcg_hr(5, buffer)
        user_ndcg10, user_hr10 = new_metrics.calculate_user_ndcg_hr(10, buffer)
        user_ndcg20, user_hr20 = new_metrics.calculate_user_ndcg_hr(20, buffer)
        # ndcg = 0

        # output_file =
        print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
              "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
              format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
                     user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20))

        print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
              "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
              format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
                     user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20), file=writer)


def predict_per_group(predict_dataset, model_obj, device, args, train_epoch, train_step, writer=None):
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
        # print(batch_data)
        logits = model_obj({
            key: value.to(device)
            for key, value in batch_data.items()
            # if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL, consts.FIELD_TRIGGER_SEQUENCE}
            if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
        })

        prob = torch.sigmoid(logits).detach().cpu().numpy()
        y = batch_data[consts.FIELD_LABEL].view(-1, 1)

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

    return pred_list, y_list, buffer


# def train(train_dataset, model_obj, device, args, bucket, pred_dataloader):
# def train(train_dataset, model_obj, device, args, pred_dataloader):
def train(train_dataset, model_obj, device, args, predict_dataloader_list):

    model_obj.to(device)

    print(model_obj)

    logger.info("Start training...")

    y_list_overall = []
    pred_list_overall = []
    buffer_overall = []

    for group_index in range(args.group_num):
        y_list_group = []
        pred_list_group = []
        buffer_group = []
        pred_dataloader = predict_dataloader_list[group_index]

        pred_list, y_list, buffer = predict_per_group(
            predict_dataset=pred_dataloader,
            model_obj=model_obj,
            device=device,
            args=args,
            # bucket=bucket,
            # train_epoch=epoch,
            train_epoch=0,
            # train_step=epoch * max_step,
            train_step=0,
            # writer=writer
        )
        pred_list_overall.extend(pred_list)
        y_list_overall.extend(y_list)
        buffer_overall.extend(buffer)
        overall_auc, _, _, _ = new_metrics.calculate_overall_auc(np.array(pred_list), np.array(y_list))
        user_auc = new_metrics.calculate_user_auc(buffer)
        # overall_logloss = new_metrics.calculate_overall_logloss(np.array(pred_list), np.array(y_list))
        # user_ndcg, user_hr = new_metrics.calculate_user_ndcg_hr(10, np.array(pred_list), np.array(y_list))
        user_ndcg5, user_hr5 = new_metrics.calculate_user_ndcg_hr(5, buffer)
        user_ndcg10, user_hr10 = new_metrics.calculate_user_ndcg_hr(10, buffer)
        user_ndcg20, user_hr20 = new_metrics.calculate_user_ndcg_hr(20, buffer)
        # with open(os.path.join(args.save_dir, "group_{}_test.txt".format(args.group_num)), "a") as writer:
        with open(os.path.join(args.checkpoint_dir, "group_{}_test.txt".format(args.group_num)), "a") as writer:
            # print(group_index, overall_auc, user_auc, overall_logloss,
            print(group_index, overall_auc, user_auc,
                  user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20, sep="\t", file=writer)

    overall_auc, _, _, _ = new_metrics.calculate_overall_auc(np.array(pred_list_overall), np.array(y_list_overall))
    user_auc = new_metrics.calculate_user_auc(buffer_overall)
    # overall_logloss = new_metrics.calculate_overall_logloss(np.array(pred_list_overall), np.array(y_list_overall))
    # user_ndcg, user_hr = new_metrics.calculate_user_ndcg_hr(10, np.array(pred_list), np.array(y_list))
    user_ndcg5, user_hr5 = new_metrics.calculate_user_ndcg_hr(5, buffer_overall)
    user_ndcg10, user_hr10 = new_metrics.calculate_user_ndcg_hr(10, buffer_overall)
    user_ndcg20, user_hr20 = new_metrics.calculate_user_ndcg_hr(20, buffer_overall)
    # with open(os.path.join(args.save_dir, "test.txt"), "a") as writer:
    with open(os.path.join(args.checkpoint_dir, "test.txt"), "a") as writer:
        # print(args.group_num, overall_auc, user_auc, overall_logloss,
        print(args.group_num, overall_auc, user_auc,
              user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20, sep="\t", file=writer)


def load_model(args, model_obj):
    ckpt_path = os.path.join(args.checkpoint_dir, "best_auc.pkl")
    # output_file = os.path.join(args.checkpoint_dir, "test.txt")
    # feature_file = os.path.join(args.checkpoint_dir, "feature.pt")

    model_load = torch.load(ckpt_path)
    model_load_name_set = set()
    for name, parms in model_load.named_parameters():
        model_load_name_set.add(name)
    model_load_dict = model_load.state_dict()
    model_obj_dict = model_obj.state_dict()
    model_obj_dict.update(model_load_dict)
    model_obj.load_state_dict(model_obj_dict)
    # for name, parms in model_obj.named_parameters():
    #     if name in model_load_name_set:
    #         # print("-" * 50)
    #         # print(name)
    #         # print(parms)
    #         parms.requires_grad = False
    return model_obj, model_load_name_set


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

    # Load model configuration
    # model_conf, raw_model_conf = ap.parse_arch_config_from_args(model_meta, args, bucket)  # type: dict
    model_conf, raw_model_conf = ap.parse_arch_config_from_args(model_meta, args)  # type: dict

    # Dump arguments and model architecture configuration to OSS
    # ap.dump_train_arguments(args.checkpoint_dir, args, bucket)
    # ap.dump_model_config(args.checkpoint_dir, raw_model_conf, bucket)

    # Construct model
    model_obj = model_meta.model_builder(model_conf=model_conf)  # type: torch.nn.module
    # model_obj, model_load_name_set = load_model(args, model_obj)

    # print("=" * 100)
    # for name, parms in model_obj.named_parameters():
    #     print(name)
    # print("=" * 100)
    device = env.get_device()
    # worker_id, worker_count = env.get_cluster_info()
    worker_id = worker_count = 8
    train_file, test_file = args.dataset.split(',')
    args.train_ood_threshold = 0.5
    # important!!!!
    args.num_loading_workers = 1

    # Setup up data loader
    pred_dataloader = meta_sequence_dataloader.MetaSequenceDataLoader(
        table_name=test_file,
        slice_id=args.num_loading_workers * worker_id,
        slice_count=args.num_loading_workers * worker_count,
        is_train=False
    )
    pred_dataloader = torch.utils.data.DataLoader(
        pred_dataloader,
        batch_size=args.batch_size,
        # batch_size=1,
        num_workers=args.num_loading_workers,
        pin_memory=True,
        collate_fn=pred_dataloader.batchify,
        drop_last=False
    )

    best_model = None
    folder = args.checkpoint_dir.split("/")[-1]
    # if folder == "meta_ood":
    #     best_model = torch.load(os.path.join(args.checkpoint_dir, "best_auc_ood.pkl"))
    # elif folder == "meta" or folder == "meta_random" or folder == "meta_ood_lof" or folder == "meta_ood_ocsvm":
    #     best_model = torch.load(os.path.join(args.checkpoint_dir, "best_auc.pkl"))
    # elif folder == "meta_ood_uncertainty":
    #     best_model = torch.load(os.path.join(args.checkpoint_dir, "stage3_best_auc_ood.pkl"))
    best_model = torch.load(os.path.join(args.checkpoint_dir, "best_auc.pkl"))

    epoch = 0
    max_step = 0
    # pred_overall_auc, pred_user_auc, pred_overall_logloss, pred_user_ndcg5, pred_user_hr5, \
    # pred_user_ndcg10, pred_user_hr10, pred_user_ndcg20, pred_user_hr20, pred_ood_auc, \
    # _threshold, _ood_threshold = \
    predict(
        predict_dataset=pred_dataloader,
        model_obj=model_obj,
        device=device,
        args=args,
        # bucket=bucket,
        train_epoch=epoch,
        train_step=epoch * max_step,
        # writer=writer
        epoch=epoch,
        # train_ood_threshold=train_ood_threshold,
        best_model=best_model
    )

    # # finetune_dataset_dir = os.path.join(args.dataset, "finetune_dataset")
    # finetune_dataset_dir = os.path.join(args.dataset_folder, "finetune_dataset")
    # print(model_obj)
    # # user_folder_name_list = os.listdir(finetune_dataset_dir)[:50]
    # # user_folder_name_list = os.listdir(finetune_dataset_dir)[22]
    # # user_folder_name_list = os.listdir(finetune_dataset_dir)[:5]
    # model_obj = torch.load(os.path.join(args.checkpoint_dir, "best_auc.pkl"))
    # user_folder_name_list = os.listdir(finetune_dataset_dir)
    # for index, user_folder_name in enumerate(user_folder_name_list, 1):
    #     print(index, len(user_folder_name_list), sep="/")
    #     # train_file, test_file = args.dataset.split(',')
    #     # model_obj = copy.deepcopy(model_obj_backup)
    #     # train_file, test_file = os.listdir(os.path.join(finetune_dataset_dir, user_folder_name))
    #     train_file = os.path.join(finetune_dataset_dir, user_folder_name, "train.txt")
    #     # if not os.path.exists(train_file):
    #     #     is_finetune = False
    #     # else:
    #     #     is_finetune = True
    #     test_file = os.path.join(finetune_dataset_dir, user_folder_name, "test.txt")
    #     # print("*" * 50)
    #     # print(os.listdir(os.path.join(finetune_dataset_dir, user_folder_name)))
    #     # print(train_file, test_file)
    #     # train_file = os.path.join(os.path.join(finetune_dataset_dir, user_folder_name, train_file))
    #     # test_file = os.path.join(os.path.join(finetune_dataset_dir, user_folder_name, test_file))
    #
    #     # important!!!!
    #     args.num_loading_workers = 1
    #     # Setup up data loader
    #     # train_dataloader = sequence_dataloader.SequenceDataLoader(
    #     # Setup up data loader
    #     # pred_dataloader = sequence_dataloader.SequenceDataLoader(
    #     pred_dataloader = sequence_dataloader.MetaSequenceDataLoader(
    #         table_name=test_file,
    #         slice_id=args.num_loading_workers * worker_id,
    #         slice_count=args.num_loading_workers * worker_count,
    #         is_train=False
    #     )
    #     pred_dataloader = torch.utils.data.DataLoader(
    #         dataset=pred_dataloader,
    #         batch_size=args.batch_size,
    #         num_workers=args.num_loading_workers,
    #         pin_memory=True,
    #         collate_fn=pred_dataloader.batchify,
    #         drop_last=False,
    #         # shuffle=True
    #     )
    #     predict_per_user(
    #         predict_dataset=pred_dataloader,
    #         model_obj=model_obj,
    #         device=device,
    #         args=args,
    #         # bucket=bucket,
    #         train_epoch=epoch,
    #         train_step=epoch * max_step,
    #         # writer=writer
    #         epoch=epoch,
    #         # train_ood_threshold=train_ood_threshold,
    #         best_model=best_model
    #     )
    model_obj = torch.load(os.path.join(args.checkpoint_dir, "best_auc.pkl"))
    dataset_folder = os.path.join(args.dataset_folder, "domain_label_grad_gru_random_{}_final".format(random_seed),
                                  # args.model, str(args.group_num))
                                  args.base_model, str(args.group_num))
    # args.base_model, str(args.group_num))
    # model_obj = torch.load(os.path.join(args.checkpoint_dir, "best_auc.pkl"))
    pred_dataloader_list = []
    for group_index in range(args.group_num):
        test_file = os.path.join(dataset_folder, "test_{}.txt".format(group_index))
        print(test_file)
        # Setup up data loader
        # pred_dataloader = sequence_dataloader.SequenceDataLoader(
        # pred_dataloader = sequence_dataloader.MetaSequenceDataLoader(
        pred_dataloader = meta_sequence_dataloader.MetaSequenceDataLoader(
            table_name=test_file,
            slice_id=args.num_loading_workers * worker_id,
            slice_count=args.num_loading_workers * worker_count,
            is_train=False
        )
        pred_dataloader = torch.utils.data.DataLoader(
            dataset=pred_dataloader,
            batch_size=args.batch_size,
            num_workers=args.num_loading_workers,
            pin_memory=True,
            collate_fn=pred_dataloader.batchify,
            drop_last=False,
            # shuffle=True
        )
        pred_dataloader_list.append(pred_dataloader)
    train_dataloader = None
    # Setup training
    train(
        train_dataset=train_dataloader,
        model_obj=model_obj,
        device=device,
        args=args,
        # bucket=bucket,
        # pred_dataloader=pred_dataloader
        predict_dataloader_list=pred_dataloader_list
    )

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    mp.spawn(main_worker, nprocs=1)
