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
    parser.add_argument("--pretrain_model_path", type=str, help="Path of the checkpoint path")
    parser.add_argument('--pretrain', '-p', action='store_true', help='load pretrained model')

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

    return parser.parse_known_args()[0]


def get_all_user_set(predict_dataset):
    user_list = []
    for step, batch_data in tqdm(enumerate(predict_dataset, 1)):
        for key, value in batch_data.items():
            # if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
            # print(key)
            if key == consts.FIELD_USER_ID:
                user_list.extend(value.numpy().flatten())
    return set(user_list)


# def predict(predict_dataset, model_obj, device, args, bucket, train_step, writer=None):
# def predict(predict_dataset, model_obj, device, args, train_epoch, train_step, writer=None, epoch=0,
def predict(train_file, test_file, model_obj, device, args, train_epoch, train_step, writer=None, epoch=0,
            train_ood_threshold=0.5, best_model=None, pretrain_model_param_dict=None):

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
    # WRITING_BATCH_SIZE = 512
    #
    # ood_prob_list = []
    # for step, batch_data in tqdm(enumerate(predict_dataset, 1)):
    #     logits, ood_logits, request_num, total_num = model_obj({
    #         key: value.to(device)
    #         for key, value in batch_data.items()
    #         if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
    #         # }, pred=True, train_ood_threshold=train_ood_threshold)
    #     # }, pred=True, train_ood_threshold=args.train_ood_threshold, use_uncertainty_net=True, stage=3)
    #     }, pred=True, mis_rec_threshold=args.train_ood_threshold, use_uncertainty_net=True, stage=3)
    #     prob = torch.sigmoid(logits).detach().cpu().numpy()
    #     y = batch_data[consts.FIELD_LABEL].view(-1, 1)
    #     ood_prob = torch.sigmoid(ood_logits).detach().cpu().numpy()
    #     ood_prob_list.extend(ood_prob)
    worker_id = worker_count = 8

    for index, (mode, file) in enumerate(zip(["train", "test"], [train_file, test_file])):
        _grad_list1 = []
        _grad_list2 = []
        _grad_list3 = []
        # Setup up data loader
        pred_dataloader = meta_sequence_dataloader.MetaSequenceDataLoader(
            # table_name=test_file,
            table_name=file,
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
        buffer = []
        pred_list = []
        y_list = []
        user_id_list = []
        # hist_embed_list = []

        base_model_name = args.model.split("_")[1]
        for step, batch_data in tqdm(enumerate(pred_dataloader, 1)):
            logits, grad_list = model_obj({
                key: value.to(device)
                for key, value in batch_data.items()
                if key not in {consts.FIELD_USER_ID, consts.FIELD_LABEL}
            }, pretrain_model=pretrain_model_param_dict, return_grad=True)
            # _grad_list1.extend(grad_list[0].detach().cpu())
            # _grad_list2.extend(grad_list[1].detach().cpu())
            # _grad_list3.extend(grad_list[2].detach().cpu())
            # print(args.model)
            # base_model_name = args.model.split("_")[1]
            # if args.model == "meta_din_prototype_grad":
            if base_model_name == "din":
                # if grad_list[-1].detach().cpu() not in _grad_list1 \
                #         and grad_list[-2].detach().cpu() not in _grad_list2 \
                #         and grad_list[-3].detach().cpu() not in _grad_list3:
                    _grad_list1.extend(grad_list[-1].detach().cpu())
                    _grad_list2.extend(grad_list[-2].detach().cpu())
                    _grad_list3.extend(grad_list[-3].detach().cpu())
            # elif args.model == "meta_gru4rec_prototype_grad" or args.model == "meta_sasrec_prototype_grad":
            elif base_model_name == "gru4rec" or base_model_name == "sasrec":
                # if grad_list[-1].detach().cpu() not in _grad_list1 \
                #         and grad_list[-2].detach().cpu() not in _grad_list2:
                    _grad_list1.extend(grad_list[-1].detach().cpu())
                    _grad_list2.extend(grad_list[-2].detach().cpu())
            # print(len(grad_list))
            prob = torch.sigmoid(logits).detach().cpu().numpy()
            y = batch_data[consts.FIELD_LABEL].view(-1, 1)
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
        with open(os.path.join(args.checkpoint_dir, "test.txt"), "a") as writer:
            print("train_epoch={}, train_step={}, overall_auc={:5f}, user_auc={:5f}, overall_logloss={:5f}, "
                  "user_ndcg5={:5f}, user_hr5={:5f}, user_ndcg10={:5f}, user_hr10={:5f}, user_ndcg20={:5f}, user_hr20={:5f}".
                  format(train_epoch, train_step, overall_auc, user_auc, overall_logloss,
                         user_ndcg5, user_hr5, user_ndcg10, user_hr10, user_ndcg20, user_hr20), file=writer)

        args.grad_list1_pt_path = os.path.join(args.checkpoint_dir, "{}_grad_list1.pt".format(mode))
        args.grad_list2_pt_path = os.path.join(args.checkpoint_dir, "{}_grad_list2.pt".format(mode))
        args.grad_list3_pt_path = os.path.join(args.checkpoint_dir, "{}_grad_list3.pt".format(mode))
        # if args.model == "din":
        if base_model_name == "din":
            torch.save(_grad_list1, args.grad_list1_pt_path)
            torch.save(_grad_list2, args.grad_list2_pt_path)
            torch.save(_grad_list3, args.grad_list3_pt_path)
        # elif args.model == "gru4rec" or args.model == "sasrec":
        elif base_model_name == "gru4rec" or base_model_name == "sasrec":
            torch.save(_grad_list1, args.grad_list1_pt_path)
            torch.save(_grad_list2, args.grad_list2_pt_path)


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
    # model_meta = model.get_model_meta(args.model_uncertainty)  # type: model.ModelMeta

    # args.grad_list1_pt_path = os.path.join(args.checkpoint_dir, "grad_list1.pt")
    # args.grad_list2_pt_path = os.path.join(args.checkpoint_dir, "grad_list2.pt")
    # args.grad_list3_pt_path = os.path.join(args.checkpoint_dir, "grad_list3.pt")

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

    if args.pretrain:
        ckpt = torch.load(args.pretrain_model_path, map_location='cpu')
        ckpt.to(device)
        # weight = ckpt["net"]
        trunk_weight = {}
        # import ipdb
        # ipdb.set_trace()
        # param_dict = net.state_dict()
        pretrain_model_param_dict = ckpt.state_dict()
        # finetune_model_keys = set(list(net.state_dict().keys()))
        # pretrain_model_keys_set = weight.keys()
        # for index, name in enumerate(pretrain_model_keys_set):
        #     if name in finetune_model_keys:
        #         param_dict[name] = weight[name]
        # print(ckpt)
    else:
        pretrain_model_param_dict = None

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
    #     # batch_size=1,
    #     num_workers=args.num_loading_workers,
    #     pin_memory=True,
    #     collate_fn=pred_dataloader.batchify,
    #     drop_last=False
    # )

    # best_model = None
    # folder = args.checkpoint_dir.split("/")[-1]
    # if folder == "meta_ood":
    #     best_model = torch.load(os.path.join(args.checkpoint_dir, "best_auc_ood.pkl"))
    # elif folder == "meta_random" or folder == "meta_ood_lof" or folder == "meta_ood_ocsvm":
    #     best_model = torch.load(os.path.join(args.checkpoint_dir, "best_auc.pkl"))
    # elif folder == "meta_ood_uncertainty" or folder == "meta_ood_uncertainty5" or folder == "meta_ood_uncertainty5_2"\
    #         or folder == "meta_ood_uncertainty_apg" or folder == "meta_ood_uncertainty5_apg" or folder == "meta_ood_uncertainty5_2_apg":
    #     best_model = torch.load(os.path.join(args.checkpoint_dir, "stage3_best_auc_ood.pkl"))
    best_model = torch.load(os.path.join(args.checkpoint_dir, "best_auc.pkl"))
    epoch = 0
    max_step = 0
    # pred_overall_auc, pred_user_auc, pred_overall_logloss, pred_user_ndcg5, pred_user_hr5, \
    # pred_user_ndcg10, pred_user_hr10, pred_user_ndcg20, pred_user_hr20, pred_ood_auc, \
    # _threshold, _ood_threshold = \
    predict(
        # predict_dataset=pred_dataloader,
        train_file=train_file,
        test_file=test_file,
        model_obj=model_obj,
        device=device,
        args=args,
        # bucket=bucket,
        train_epoch=epoch,
        train_step=epoch * max_step,
        # writer=writer
        epoch=epoch,
        # train_ood_threshold=train_ood_threshold,
        best_model=best_model,
        pretrain_model_param_dict=pretrain_model_param_dict
    )


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    mp.spawn(main_worker, nprocs=1)
