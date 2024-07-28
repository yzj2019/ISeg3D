"""
Tester

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import time
import numpy as np
from collections import OrderedDict
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data

from .defaults import create_ddp_model
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry
from pointcept.utils.misc import (
    AverageMeter,
    intersection_and_union,
    intersection_and_union_gpu,
    make_dirs,
)

from pointcept.datasets.utils import collate_fn
import pointcept.utils.comm as comm
from pointcept.utils.clicker import CLICKER
from pointcept.utils.merger import ClicksMerger
from pointcept.utils.matcher import HungarianMatcher
import pointops

TESTERS = Registry("testers")


class TesterBase:
    def __init__(self, cfg, model=None, test_loader=None, verbose=False) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "test.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.verbose = verbose
        if self.verbose:
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.test_loader = self.build_test_loader()
        else:
            self.test_loader = test_loader

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint["epoch"]
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def build_test_loader(self):
        test_dataset = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=self.cfg.batch_size_test_per_gpu,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
        )
        return test_loader

    def test(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise collate_fn(batch)


@TESTERS.register_module()
class SemSegTester(TesterBase):
    def test(self):
        assert self.test_loader.batch_size == 1
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        save_path = os.path.join(self.cfg.save_path, "result")
        make_dirs(save_path)
        # create submit folder only on main process
        if (
            self.cfg.data.test.type == "ScanNetDataset"
            or self.cfg.data.test.type == "ScanNet200Dataset"
        ) and comm.is_main_process():
            make_dirs(os.path.join(save_path, "submit"))
        elif (
            self.cfg.data.test.type == "SemanticKITTIDataset" and comm.is_main_process()
        ):
            make_dirs(os.path.join(save_path, "submit"))
        elif self.cfg.data.test.type == "NuScenesDataset" and comm.is_main_process():
            import json

            make_dirs(os.path.join(save_path, "submit", "lidarseg", "test"))
            make_dirs(os.path.join(save_path, "submit", "test"))
            submission = dict(
                meta=dict(
                    use_camera=False,
                    use_lidar=True,
                    use_radar=False,
                    use_map=False,
                    use_external=False,
                )
            )
            with open(
                os.path.join(save_path, "submit", "test", "submission.json"), "w"
            ) as f:
                json.dump(submission, f, indent=4)
        comm.synchronize()
        record = {}
        # fragment inference
        for idx, data_dict in enumerate(self.test_loader):
            end = time.time()
            data_dict = data_dict[0]  # current assume batch size is 1
            fragment_list = data_dict.pop("fragment_list")
            segment = data_dict.pop("segment")
            data_name = data_dict.pop("name")
            pred_save_path = os.path.join(save_path, "{}_pred.npy".format(data_name))
            if os.path.isfile(pred_save_path):
                logger.info(
                    "{}/{}: {}, loaded pred and label.".format(
                        idx + 1, len(self.test_loader), data_name
                    )
                )
                pred = np.load(pred_save_path)
            else:
                pred = torch.zeros((segment.size, self.cfg.data.num_classes)).cuda()
                for i in range(len(fragment_list)):
                    fragment_batch_size = 1
                    s_i, e_i = i * fragment_batch_size, min(
                        (i + 1) * fragment_batch_size, len(fragment_list)
                    )
                    input_dict = collate_fn(fragment_list[s_i:e_i])
                    for key in input_dict.keys():
                        if isinstance(input_dict[key], torch.Tensor):
                            input_dict[key] = input_dict[key].cuda(non_blocking=True)
                    idx_part = input_dict["index"]
                    with torch.no_grad():
                        pred_part = self.model(input_dict)["seg_logits"]  # (n, k)
                        pred_part = F.softmax(pred_part, -1)
                        if self.cfg.empty_cache:
                            torch.cuda.empty_cache()
                        bs = 0
                        for be in input_dict["offset"]:
                            pred[idx_part[bs:be], :] += pred_part[bs:be]
                            bs = be

                    logger.info(
                        "Test: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}".format(
                            idx + 1,
                            len(self.test_loader),
                            data_name=data_name,
                            batch_idx=i,
                            batch_num=len(fragment_list),
                        )
                    )
                pred = pred.max(1)[1].data.cpu().numpy()
                np.save(pred_save_path, pred)
            if "origin_segment" in data_dict.keys():
                assert "inverse" in data_dict.keys()
                pred = pred[data_dict["inverse"]]
                segment = data_dict["origin_segment"]
            intersection, union, target = intersection_and_union(
                pred, segment, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)
            record[data_name] = dict(
                intersection=intersection, union=union, target=target
            )

            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])
            acc = sum(intersection) / (sum(target) + 1e-10)

            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}]-{} "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {acc:.4f} ({m_acc:.4f}) "
                "mIoU {iou:.4f} ({m_iou:.4f})".format(
                    data_name,
                    idx + 1,
                    len(self.test_loader),
                    segment.size,
                    batch_time=batch_time,
                    acc=acc,
                    m_acc=m_acc,
                    iou=iou,
                    m_iou=m_iou,
                )
            )
            if (
                self.cfg.data.test.type == "ScanNetDataset"
                or self.cfg.data.test.type == "ScanNet200Dataset"
            ):
                np.savetxt(
                    os.path.join(save_path, "submit", "{}.txt".format(data_name)),
                    self.test_loader.dataset.class2id[pred].reshape([-1, 1]),
                    fmt="%d",
                )
            elif self.cfg.data.test.type == "SemanticKITTIDataset":
                # 00_000000 -> 00, 000000
                sequence_name, frame_name = data_name.split("_")
                os.makedirs(
                    os.path.join(
                        save_path, "submit", "sequences", sequence_name, "predictions"
                    ),
                    exist_ok=True,
                )
                pred = pred.astype(np.uint32)
                pred = np.vectorize(
                    self.test_loader.dataset.learning_map_inv.__getitem__
                )(pred).astype(np.uint32)
                pred.tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "sequences",
                        sequence_name,
                        "predictions",
                        f"{frame_name}.label",
                    )
                )
            elif self.cfg.data.test.type == "NuScenesDataset":
                np.array(pred + 1).astype(np.uint8).tofile(
                    os.path.join(
                        save_path,
                        "submit",
                        "lidarseg",
                        "test",
                        "{}_lidarseg.bin".format(data_name),
                    )
                )

        logger.info("Syncing ...")
        comm.synchronize()
        record_sync = comm.gather(record, dst=0)

        if comm.is_main_process():
            record = {}
            for _ in range(len(record_sync)):
                r = record_sync.pop()
                record.update(r)
                del r
            intersection = np.sum(
                [meters["intersection"] for _, meters in record.items()], axis=0
            )
            union = np.sum([meters["union"] for _, meters in record.items()], axis=0)
            target = np.sum([meters["target"] for _, meters in record.items()], axis=0)

            if self.cfg.data.test.type == "S3DISDataset":
                torch.save(
                    dict(intersection=intersection, union=union, target=target),
                    os.path.join(save_path, f"{self.test_loader.dataset.split}.pth"),
                )

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info(
                "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}".format(
                    mIoU, mAcc, allAcc
                )
            )
            for i in range(self.cfg.data.num_classes):
                logger.info(
                    "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                        idx=i,
                        name=self.cfg.data.names[i],
                        iou=iou_class[i],
                        accuracy=accuracy_class[i],
                    )
                )
            logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return batch


@TESTERS.register_module()
class ClsTester(TesterBase):
    def test(self):
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        self.model.eval()

        for i, input_dict in enumerate(self.test_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            end = time.time()
            with torch.no_grad():
                output_dict = self.model(input_dict)
            output = output_dict["cls_logits"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = intersection_and_union_gpu(
                pred, label, self.cfg.data.num_classes, self.cfg.data.ignore_index
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(
                    target
                )
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            intersection_meter.update(intersection), union_meter.update(
                union
            ), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)

            logger.info(
                "Test: [{}/{}] "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Accuracy {accuracy:.4f} ".format(
                    i + 1,
                    len(self.test_loader),
                    batch_time=batch_time,
                    accuracy=accuracy,
                )
            )

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
        logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                mIoU, mAcc, allAcc
            )
        )

        for i in range(self.cfg.data.num_classes):
            logger.info(
                "Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=accuracy_class[i],
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


@TESTERS.register_module()
class PartSegTester(TesterBase):
    def test(self):
        test_dataset = self.test_loader.dataset
        logger = get_root_logger()
        logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        batch_time = AverageMeter()

        num_categories = len(self.test_loader.dataset.categories)
        iou_category, iou_count = np.zeros(num_categories), np.zeros(num_categories)
        self.model.eval()

        save_path = os.path.join(
            self.cfg.save_path, "result", "test_epoch{}".format(self.cfg.test_epoch)
        )
        make_dirs(save_path)

        for idx in range(len(test_dataset)):
            end = time.time()
            data_name = test_dataset.get_data_name(idx)

            data_dict_list, label = test_dataset[idx]
            pred = torch.zeros((label.size, self.cfg.data.num_classes)).cuda()
            batch_num = int(np.ceil(len(data_dict_list) / self.cfg.batch_size_test))
            for i in range(batch_num):
                s_i, e_i = i * self.cfg.batch_size_test, min(
                    (i + 1) * self.cfg.batch_size_test, len(data_dict_list)
                )
                input_dict = collate_fn(data_dict_list[s_i:e_i])
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                with torch.no_grad():
                    pred_part = self.model(input_dict)["cls_logits"]
                    pred_part = F.softmax(pred_part, -1)
                if self.cfg.empty_cache:
                    torch.cuda.empty_cache()
                pred_part = pred_part.reshape(-1, label.size, self.cfg.data.num_classes)
                pred = pred + pred_part.total(dim=0)
                logger.info(
                    "Test: {} {}/{}, Batch: {batch_idx}/{batch_num}".format(
                        data_name,
                        idx + 1,
                        len(test_dataset),
                        batch_idx=i,
                        batch_num=batch_num,
                    )
                )
            pred = pred.max(1)[1].data.cpu().numpy()

            category_index = data_dict_list[0]["cls_token"]
            category = self.test_loader.dataset.categories[category_index]
            parts_idx = self.test_loader.dataset.category2part[category]
            parts_iou = np.zeros(len(parts_idx))
            for j, part in enumerate(parts_idx):
                if (np.sum(label == part) == 0) and (np.sum(pred == part) == 0):
                    parts_iou[j] = 1.0
                else:
                    i = (label == part) & (pred == part)
                    u = (label == part) | (pred == part)
                    parts_iou[j] = np.sum(i) / (np.sum(u) + 1e-10)
            iou_category[category_index] += parts_iou.mean()
            iou_count[category_index] += 1

            batch_time.update(time.time() - end)
            logger.info(
                "Test: {} [{}/{}] "
                "Batch {batch_time.val:.3f} "
                "({batch_time.avg:.3f}) ".format(
                    data_name, idx + 1, len(self.test_loader), batch_time=batch_time
                )
            )

        ins_mIoU = iou_category.sum() / (iou_count.sum() + 1e-10)
        cat_mIoU = (iou_category / (iou_count + 1e-10)).mean()
        logger.info(
            "Val result: ins.mIoU/cat.mIoU {:.4f}/{:.4f}.".format(ins_mIoU, cat_mIoU)
        )
        for i in range(num_categories):
            logger.info(
                "Class_{idx}-{name} Result: iou_cat/num_sample {iou_cat:.4f}/{iou_count:.4f}".format(
                    idx=i,
                    name=self.test_loader.dataset.categories[i],
                    iou_cat=iou_category[i] / (iou_count[i] + 1e-10),
                    iou_count=int(iou_count[i]),
                )
            )
        logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


# modification
    
@TESTERS.register_module()
class InsSegTester(TesterBase):
    """
    Instance Segmentation Tester
    """
    def __init__(
        self, cfg, model=None, test_loader=None, verbose=False,
        semantic_ignore_label=[-1], instance_ignore_label=-1,
        topk_per_scene=100, cost_class=1.0, cost_focal=1.0, cost_dice=1.0,
    ) -> None:
        super().__init__(cfg, model, test_loader, verbose)
        self.semantic_ignore_label = semantic_ignore_label
        self.instance_ignore_label = instance_ignore_label
        self.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.topk_per_scene = topk_per_scene        # 按 cls_score 取/复制 topk 个 clicks 作为 instance
        self.matcher = HungarianMatcher(
            cost_class=cost_class, cost_focal=cost_focal, cost_dice=cost_dice,
            instance_ignore_label=instance_ignore_label
        )
    
    def get_target(self, semantic, instance):
        '''
        reconstruct the ground truth

        Returns:
        - ins_classes: [N_target_instance], ins_idx -> ins_tag
        - masks_target: [N_points x N_target_instance], 一个点不一定只对应一个click
        - cls_target: [N_target_instance], ins_idx -> sem_tag
        - is_testset: if testing on test set
        '''
        ins_classes = torch.unique(instance[instance != self.instance_ignore_label]).int()
        masks_target = torch.zeros((semantic.shape[0], len(ins_classes))).to(semantic.device).float()
        cls_target = torch.zeros((len(ins_classes),)).to(semantic.device).int()
        is_testset = len(ins_classes) == 0
        for i in range(len(ins_classes)):
            class_tag = ins_classes[i]
            ins_filter = (instance == class_tag)
            masks_target[:,i][ins_filter] = 1.0
            cls_target[i] = semantic[ins_filter][0]
        return ins_classes, masks_target, cls_target, is_testset


    def get_pred(self, masks_heatmap:torch.Tensor, cls_logits:torch.Tensor):
        '''
        modified from mask3D, before match and after merge, reconstruct prediction
        https://github.com/JonasSchult/Mask3D/blob/main/trainer/trainer.py
        - masks_heatmap: N_points x N_clicks, as heatmap
        - cls_logits: N_clicks x num_classes

        Returns:
        - score: tensor, [N_pred_instance]
        - masks_pred: int tensor and bool value, [N_points x N_pred_instance]
        - cls_pred: int tensor, [N_pred_instance]
        - masks_heatmap: tensor, [N_points x N_pred_instance]
        '''
        cls_pred = cls_logits.softmax(-1)
        num_queries = masks_heatmap.shape[1]
        num_classes = cls_logits.shape[1]

        # 减小分类错误带来的影响: flatten, 一个click query可能生成多个instance, 如果它的 cls_pred 值互相之间比较接近
        labels = (
            torch.arange(num_classes, device=masks_heatmap.device)
            .unsqueeze(0)
            .repeat(num_queries, 1)
            .flatten(0, 1)
        )

        if self.topk_per_scene < cls_logits.shape[0] * num_classes:
            cls_scores_per_query, topk_indices = cls_logits.flatten(0, 1).topk(
                self.topk_per_scene, sorted=True
            )
        else:
            cls_scores_per_query, topk_indices = cls_logits.flatten(0, 1).topk(
                num_queries, sorted=True
            )

        labels_per_query = labels[topk_indices]
        topk_indices =  torch.div(topk_indices, num_classes, rounding_mode='floor')
        masks_heatmap = masks_heatmap[:, topk_indices]

        mask = (masks_heatmap > 0.5)
        masks_heatmap[~mask] = 0            # Truncated
        masks_pred = mask.int()
        cls_pred = labels_per_query

        mask_scores_per_query = (masks_pred * masks_heatmap).sum(0) / (
            masks_heatmap.sum(0) + 1e-6
        )
        score = cls_scores_per_query * mask_scores_per_query        # scores per query
        
        return score, masks_pred, cls_pred, masks_heatmap

    
    def associate_instances(self, pred, target):
        '''匈牙利匹配, 为每个gt最优指派一个pred'''
        idx_pred, idx_target = self.matcher(pred, target)
        # print(idx_pred, idx_target)
        # print(pred["cls_pred"][idx_pred])
        # print(target["labels"][idx_target])
        pred["matched_idx"] = idx_pred
        target["matched_idx"] = idx_target
        pred["vert_count"] = pred["masks_heatmap"].sum(0)       # 面积
        target["vert_count"] = target["masks"].sum(0)
        pred["idx_to_target_idx"] = {idx_pred[i].item():idx_target[i].item() for i in range(len(idx_pred))}
        target["idx_to_pred_idx"] = {idx_target[i].item():idx_pred[i].item() for i in range(len(idx_pred))}
        # pred["cls_prob"] = pred["cls_prob"][idx_pred]
        # pred["masks_heatmap"] = pred["masks_heatmap"][:, idx_pred]
        # target["labels"] = target["labels"][idx_target]
        # target["masks"] = target["masks"][:, idx_target]
        return pred, target


    def evaluate_matched_ins(self, pred, target):
        '''
        modified from benchmark's evaluation
        https://github.com/ScanNet/ScanNet/blob/master/BenchmarkScripts/3d_evaluation/evaluate_semantic_instance.py
        https://github.com/Pointcept/Pointcept/blob/main/pointcept/engines/hooks/evaluator.py
        '''
        overlaps = self.overlaps
        # overlaps = [0.5]
        intersection = pred["masks_heatmap"].T @ target["masks"]        # [N_pred_instance, N_target_instance]
        union = pred["vert_count"].view(-1,1) + target["vert_count"].view(1,-1)
        union = union - intersection
        tgt_idx_to_matched_iou = {}
        tgt_idx_to_matched_pred = {}
        
        # results: class x overlap
        ap_table = np.zeros((len(self.valid_class_tags), len(overlaps)), float)
        for oi, overlap_th in enumerate(overlaps):
            for li, class_tag in enumerate(self.valid_class_tags):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False
                classi_pred_idx = torch.where(pred["cls_pred"] == class_tag)[0]
                classi_tgt_idx = torch.where(target["labels"] == class_tag)[0]

                if len(classi_pred_idx):
                    has_pred = True
                if len(classi_tgt_idx):
                    has_gt = True
                cur_true = np.ones(len(classi_tgt_idx))
                cur_score = np.ones(len(classi_tgt_idx)) * (-float("inf"))
                cur_match = np.zeros(len(classi_tgt_idx), dtype=bool)
                # collect matches
                for (gti, gt) in enumerate(classi_tgt_idx):
                    if gt.item() in target["idx_to_pred_idx"].keys():
                        pr = target["idx_to_pred_idx"][gt.item()]
                        # already assigned by matcher
                        ious = intersection[:, gt] / union[:, gt]
                        iou = ious[pr]
                        tgt_idx_to_matched_iou[gt] = iou
                        tgt_idx_to_matched_pred[gt] = pred["masks_pred"][:, pr]
                        # print(f'gt {gt}/{type(gt)}, pr {pr}/{type(pr)}, iou {iou}, max iou {ious.max()}')
                        if iou > overlap_th and torch.isin(pr, classi_pred_idx):
                            cur_match[gti] = True
                            cur_score[gti] = pred["score"][pr]
                            # append others as false positive
                            prs = torch.where(ious > overlap_th)[0]
                            prs = prs[torch.isin(prs, classi_pred_idx)]
                            prs = prs[prs != pr]
                            for pr in prs:
                                cur_true = np.append(cur_true, 0)
                                cur_score = np.append(cur_score, pred["score"][pr].cpu())
                                cur_match = np.append(cur_match, True)
                        else:
                            # under thershold or not same cls, so this gt keep unmatched
                            hard_false_negatives += 1
                # remove non-matched ground truth instances
                cur_true = cur_true[cur_match]
                cur_score = cur_score[cur_match]
                # collect non-matched predictions as false positive
                for pr in classi_pred_idx:
                    ious = intersection[pr, :] / union[pr, :]
                    gts = torch.where(ious > overlap_th)[0]
                    gts = gts[torch.isin(gts, classi_tgt_idx)]
                    found_gt = len(gts)>0
                    if not found_gt:
                        num_ignore = pred["void_intersection"][pr]
                        proportion_ignore = float(num_ignore) / pred["vert_count"][pr]
                        # if not ignored append false positive
                        if proportion_ignore <= overlap_th:
                            cur_true = np.append(cur_true, 0)
                            cur_score = np.append(cur_score, pred["score"][pr].cpu())
                # append to overall results
                y_true = np.append(y_true, cur_true)
                y_score = np.append(y_score, cur_score)
                # compute average precision
                if has_gt and has_pred:
                    # compute precision recall curve first
                    # sorting and cumsum
                    # print(f'class tag {class_tag}')
                    # print(f'tgt idx {classi_tgt_idx}')
                    # print(f'y score {y_score}')
                    score_arg_sort = np.argsort(y_score)
                    y_score_sorted = y_score[score_arg_sort]
                    y_true_sorted = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)
                    # unique thresholds
                    (thresholds, unique_indices) = np.unique(y_score_sorted, return_index=True)
                    num_prec_recall = len(unique_indices) + 1
                    # prepare precision recall
                    num_examples = len(y_score_sorted)
                    # https://github.com/ScanNet/ScanNet/pull/26
                    # all predictions are non-matched but also all of them are ignored and not counted as FP
                    # y_true_sorted_cumsum is empty
                    # num_true_examples = y_true_sorted_cumsum[-1]
                    num_true_examples = y_true_sorted_cumsum[-1] if len(y_true_sorted_cumsum) > 0 else 0
                    precision = np.zeros(num_prec_recall)
                    recall = np.zeros(num_prec_recall)
                    # deal with the first point
                    y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                    # deal with remaining
                    for idx_res, idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores - 1]
                        tp = num_true_examples - cumsum
                        fp = num_examples - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        p = float(tp) / (tp + fp)
                        r = float(tp) / (tp + fn)
                        precision[idx_res] = p
                        recall[idx_res] = r
                    # first point in curve is artificial
                    precision[-1] = 1.
                    recall[-1] = 0.
                    # print(precision, recall)
                    # compute average of precision-recall curve
                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                    recall_for_conv = np.append(recall_for_conv, 0.)
                    stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], "valid")
                    # integrate is now simply a dot product
                    ap_current = np.dot(precision, stepWidths)
                elif has_gt:
                    ap_current = 0.0
                else:
                    ap_current = float("nan")
                ap_table[li, oi] = ap_current
        o50 = np.where(np.isclose(self.overlaps, 0.5))
        o25 = np.where(np.isclose(self.overlaps, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.overlaps, 0.25)))
        ap_scores = dict()
        ap_scores["all_ap"] = np.nanmean(ap_table[:, oAllBut25])
        ap_scores["all_ap_50%"] = np.nanmean(ap_table[:, o50])
        ap_scores["all_ap_25%"] = np.nanmean(ap_table[:, o25])
        ap_scores["classes"] = {}
        for (li, class_tag) in enumerate(self.valid_class_tags):
            ap_scores["classes"][class_tag] = {}
            ap_scores["classes"][class_tag]["ap"] = np.average(ap_table[li, oAllBut25])
            ap_scores["classes"][class_tag]["ap50%"] = np.average(ap_table[li, o50])
            ap_scores["classes"][class_tag]["ap25%"] = np.average(ap_table[li, o25])
        return ap_scores, tgt_idx_to_matched_iou, tgt_idx_to_matched_pred



@TESTERS.register_module()
class InteractiveInsSegTester(InsSegTester):
    """InteractiveInsSegTester
    for indoor point cloud & RGBD interactive segmentation
    TODO 对 val set 输出 FP per instance, mask 覆盖率
    progressive mode 要求 model 必须有 .refine 方法
    - refine:
    """
    def __init__(
        self, cfg, model=None, test_loader=None, verbose=False,
        semantic_ignore_label=[-1], instance_ignore_label=-1,
        topk_per_scene=100, cost_class=1.0, cost_focal=1.0, cost_dice=1.0,
        progressive_mode=False, clicker_cfg=None,
        iou_thrs=[0.8, 0.85, 0.9], noc_thrs=[1,2,3,5,10,15]
    ) -> None:
        '''
        Argument
        - 
        - 暂时弃用 '# pcd_only=True, random_clicks=False,'
            - 后面用 DiffSeg的方式从 attn map 中聚, 借用 superpoint
        '''
        super.__init__(
            cfg, model, test_loader, verbose, 
            semantic_ignore_label, instance_ignore_label,
            topk_per_scene, cost_class, cost_focal, cost_dice
        )
        self.clicker_cfg = clicker_cfg
        self.progressive_mode = progressive_mode    # 迭代refine
        # self.pcd_only = pcd_only
        # self.random_clicks = random_clicks
        # assert progressive_mode * random_clicks != True, "cannot use random clicks for progressive segmentation"
        self.iou_thrs = iou_thrs
        self.noc_thrs = noc_thrs


    # https://github.com/TitorX/CFR-ICL-Interactive-Segmentation/blob/main/isegm/inference/utils.py
    def compute_noc_metric(self, ious_map, iou_thrs, ins_tag_to_sem_tag):
        '''
        根据ious, iou_thrs计算number of click
        - ious_map: ins_tag -> iou_arr
        - iou_thrs: list of threshold
        - ins_tag_to_sem_tag: ins_tag -> sem_tag
        '''
        # TODO 写不同类别的noc
        def _get_noc(iou_arr, iou_thr):
            '''iou_arr start from 0, as iou==0 when N_click==0'''
            vals = iou_arr >= iou_thr       # return the smallest idx of True value
            return np.argmax(vals) if np.any(vals) else len(iou_arr)
        nocs_arr = []
        for iou_thr in iou_thrs:
            nocs_list = []
            for ins_tag in ious_map.keys():
                noc = _get_noc(ious_map[ins_tag], iou_thr)
                nocs_list.append(noc)
            nocs_arr.append(nocs_list)
        nocs_arr = np.array(nocs_arr, dtype=int)
        return nocs_arr


    def compute_iou_metric(self, ious_map, noc_thrs, ins_tag_to_sem_tag):
        '''
        根据 ious, noc_thrs 计算 iou@N
        - ious_map: ins_tag -> iou_arr
        - noc_thrs: list of threshold
        - ins_tag_to_sem_tag: ins_tag -> sem_tag
        '''
        def _get_iou(iou_arr, noc_thr):
            '''iou_arr start from 0, as iou==0 when N_click==0'''
            if len(iou_arr)-1 > noc_thr:
                return iou_arr[noc_thr]
            else:
                return iou_arr[-1]
        ious_arr = []
        for noc_thr in noc_thrs:
            ious_list = []
            for ins_tag in ious_map.keys():
                iou = _get_iou(ious_map[ins_tag], noc_thr)
                ious_list.append(iou)
            ious_arr.append(ious_list)
        ious_arr = np.array(ious_arr, dtype=float)
        return ious_arr


    @torch.no_grad()
    def test(self):
        cfg, test_loader, model = self.cfg, self.test_loader, self.model
        assert test_loader.batch_size == 1
        test_dataset = test_loader.dataset
        logger = get_root_logger()
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation Interactive Ins Seg >>>>>>>>>>>>>>>>')

        batch_time = AverageMeter()
        all_ap_meter = AverageMeter()
        all_ap_50_meter = AverageMeter()
        all_ap_25_meter = AverageMeter()
        if self.progressive_mode:
            progressive_time = AverageMeter()
            number_of_click = AverageMeter()
            iou_per_click = AverageMeter()
        model.eval()

        save_path = os.path.join(cfg.save_path, "result", "test_epoch{}".format(cfg.test_epoch))
        make_dirs(save_path)
        # create submit folder only on main process
        if "ScanNet" in cfg.dataset_type and comm.is_main_process():
            # in 可以是 string 的包含
            sub_path = os.path.join(save_path, "submit")
            msk_path = os.path.join(sub_path, 'predicted_masks')
            make_dirs(sub_path)
            make_dirs(msk_path)
        # if 'SemanticKITTI' in cfg.dataset_type and comm.is_main_process():
        #     sub_path = os.path.join(save_path, "submit")
        #     make_dirs(sub_path)
        comm.synchronize()

        self.class_names = cfg.data.names
        self.num_classes = cfg.data.num_classes
        self.valid_class_tags = [i for i in range(self.num_classes)
                                  if i not in self.semantic_ignore_label]
        ap_class_meter_list = [AverageMeter() for _ in self.valid_class_tags]
        # fragment inference
        for data_idx, data_dict in enumerate(test_loader):
            end = time.time()
            # 获取data_dict
            pcd_dict = data_dict[0]['pcd_dict']
            if self.pcd_only:
                img_dict, fusion = None, None
            else:
                # TODO img_dict的获取, 需要修改, 改成后面从click中获取
                img_dict, fusion = test_dataset.get_img_fusion(pcd_dict)
                # 转GPU
                for d in [img_dict, fusion]:
                    for key in d.keys():
                        # collate_fn过了, dict下面一层就是tensor
                        if isinstance(d[key], torch.Tensor):
                            d[key] = d[key].cuda(non_blocking=True)
            # 取出属性
            fragment_list = pcd_dict.pop("fragment_list")   # 切片的点云
            segment = pcd_dict.pop("segment")       # mask
            semantic = torch.Tensor(segment).cuda(non_blocking=True)     # 都转到GPU上, 加速运算
            instance = pcd_dict.pop("instance")
            instance = torch.Tensor(instance).cuda(non_blocking=True)
            origin_coord = pcd_dict.pop("coord")
            origin_coord = torch.Tensor(origin_coord).cuda(non_blocking=True)
            data_name = pcd_dict.pop("name")        # scene name like 'scene0011_00'
            if not self.pcd_only:
                assert img_dict["scene_id"] == data_name

            masks_save_path = os.path.join(save_path, '{}_masks.pth'.format(data_name))
            if os.path.isfile(masks_save_path) and not self.progressive_mode:
                logger.info('{}/{}: {}, loaded pred and label.'.format(data_idx + 1, len(test_loader), data_name))
                masks = torch.load(masks_save_path)
                score = masks["score"].cuda(non_blocking=True)
                cls_pred = masks["cls_pred"].cuda(non_blocking=True)
                # masks idx -> binary masks
                masks_heatmap = torch.zeros((instance.shape[0], cls_pred.shape[0])).float().cuda(non_blocking=True)
                masks_pred = torch.zeros((instance.shape[0], cls_pred.shape[0])).int().cuda(non_blocking=True)
                assert len(masks["offset"]) == cls_pred.shape[0], "must have same number of pred instance"
                start_id = 0
                for i in range(cls_pred.shape[0]):
                    end_id = masks["offset"][i]
                    idx = masks["masks_idx"][start_id:end_id]
                    masks_heatmap[idx, i] = masks["masks_heatmap"][start_id:end_id].cuda(non_blocking=True)
                    masks_pred[idx, i] = 1
                    start_id = end_id

            else:
                clicker = CLICKER.build(self.clicker_cfg)
                clicker.set_state(fragment_list, img_dict=img_dict, fusion=fusion, mode='test',
                                  semantic=semantic if not self.random_clicks else torch.ones_like(semantic) * cfg.semantic_ignore_label, 
                                  instance=instance,
                                  clicks_from_instance=True)
                masks_heatmap = None
                cls_logits = None
                clicks = None
                pred_last_list = []
                merger = ClicksMerger(random_clicks=True)
                for i in range(len(fragment_list)):
                    # 对每个切片(不同切片，意味着对一个voxel内多个point的，会遍历point的颜色作为voxel的颜色)
                    fragment_batch_size = 1
                    s_i, e_i = i * fragment_batch_size, min((i + 1) * fragment_batch_size, len(fragment_list))
                    pcd_dict = collate_fn(fragment_list[s_i:e_i])         # 取出一个batch中的多个dict，构成一个dict
                    # 转 GPU
                    for key in pcd_dict.keys():
                        if isinstance(pcd_dict[key], torch.Tensor):
                            pcd_dict[key] = pcd_dict[key].cuda(non_blocking=True)

                    idx_part = pcd_dict["index"]      # voxelize产生的，voxel在原数据的point的index
                    pcd_dict["fragment_idx"] = i
                    input_dict = dict(
                        pcd_dict=pcd_dict, img_dict=img_dict, fusion=fusion, clicker=clicker
                    )
                    # 由于discrete_coords一致(实为fps太费时间), 因此后续click坐标全部沿用 fragment_list[0] 产生的clicks
                    if self.random_clicks and clicks != None:
                        input_dict["clicks"] = clicks

                    with torch.no_grad():
                        # TODO 不要反复做 backbone inference
                        clicks_dict = model(input_dict)
                        pred_refine = clicks_dict['pred_refine']
                        masks_heatmap_part = pred_refine["masks_heatmap"]
                        cls_logits_part = pred_refine["cls_logits"]
                        pred_last_list.append(pred_refine)
                        clicks = clicks_dict["clicks"]
                    if cfg.empty_cache:
                        torch.cuda.empty_cache()
                    # 利用fragment的pred构建成场景的pred
                    masks_heatmap_new = torch.zeros((segment.size, masks_heatmap_part.shape[1])).cuda()
                    # 用三线性插值填充pred到gridsample之前的分辨率
                    masks_heatmap_new[idx_part, :] = masks_heatmap_part
                    idx_part = idx_part.cpu()
                    idx_new = torch.where(~torch.isin(torch.arange(origin_coord.shape[0]), idx_part))[0]
                    coord = origin_coord[idx_part].clone().detach().contiguous()
                    coord_new = origin_coord[idx_new].clone().detach().contiguous()
                    offset = torch.Tensor([coord.shape[0]]).int().cuda()
                    offset_new = torch.Tensor([coord_new.shape[0]]).int().cuda()
                    masks_heatmap_new[idx_new, :] = pointops.interpolation(coord.float(), coord_new.float(), 
                                                                           masks_heatmap_part, offset, offset_new)
                    masks_heatmap_new[masks_heatmap_new<0] = 0          # 奇怪的问题，三线性插值出负值
                    cls_logits_new = cls_logits_part
                    if masks_heatmap == None:
                        masks_heatmap = masks_heatmap_new
                        cls_logits = cls_logits_new
                    else:
                        # 融合不同fragment的instance pred
                        masks_heatmap = torch.cat([masks_heatmap, masks_heatmap_new], dim=1)
                        cls_logits = torch.cat([cls_logits, cls_logits_new], dim=0)
                    _, masks_heatmap, cls_logits = merger(masks_heatmap, cls_logits, [])
                    logger.info(f'Test initial click: {data_idx + 1}/{len(test_loader)}-{data_name}, Batch: {i}/{len(fragment_list)}')

                score, masks_pred, cls_pred, masks_heatmap = self.get_pred(masks_heatmap, cls_logits)
                # binary masks -> idx, to reduce storage space
                masks = dict(
                    offset = np.empty(cls_pred.shape[0], dtype=int),        # N_pred_ins
                    masks_idx = torch.Tensor([]).int(),                     # all masks index
                    masks_heatmap = torch.Tensor([]),
                    score = score.cpu(),
                    cls_pred = cls_pred.cpu()
                )
                cnt = 0
                for i in range(cls_pred.shape[0]):
                    idx = torch.where(masks_pred[:, i].bool())[0].cpu()
                    cnt += len(idx)
                    masks["offset"][i] = cnt
                    masks["masks_idx"] = torch.cat([masks["masks_idx"], idx])
                    masks["masks_heatmap"] = torch.cat([masks["masks_heatmap"], masks_heatmap[idx, i].cpu()])
                # 保存结果
                torch.save(masks, masks_save_path)
            
            # reconstruct pred & target
            cls_prob = torch.zeros((cls_pred.shape[0], self.num_classes)).to(masks_heatmap.device)    # [N_pred_ins, num_classes]
            for i in range(cls_pred.shape[0]):
                ins_tag = cls_pred[i]
                cls_prob[i, ins_tag] = 1.0
            ins_classes, masks_target, cls_target, is_testset = self.get_target(semantic, instance)
            # test set, save the result to submit
            if is_testset:
                assert self.random_clicks, "test set must use random clicks"
                batch_time.update(time.time() - end)
                logger.info('Test initial click: {} [{}/{}]-{} '
                        'Batch time {batch_time.val:.3f} ({batch_time.avg:.3f}) '.format(
                            data_name, data_idx + 1, len(test_loader), segment.size,
                            batch_time=batch_time
                        ))
                if "ScanNet" in cfg.dataset_type:
                    # https://kaldir.vc.in.tum.de/scannet_benchmark/documentation#format-instance3d
                    out_cls_str_list = []
                    for i in range(cls_pred.shape[0]):
                        suffix_i = os.path.join('predicted_masks', f'{data_name}_'+f'{i}'.zfill(3)+'.txt')      # predicted_masks/scene0707_00_012.txt
                        mask_path_i = os.path.join(save_path, "submit", suffix_i)
                        np.savetxt(mask_path_i, masks_pred[:, i].int().cpu().numpy(), fmt='%d')
                        out_cls_str = suffix_i + f' {int(test_dataset.class2id[cls_pred[i]])} {float(score[i]):.4f}'
                        out_cls_str_list.append(out_cls_str)
                    cls_path = os.path.join(save_path, "submit", f'{data_name}.txt')
                    fp = open(cls_path, 'w')
                    fp.writelines(out_cls_str_list)
                    fp.close()
                continue

            # val set, match & select
            pred = dict(cls_prob=cls_prob, masks_heatmap=masks_heatmap, masks_pred=masks_pred,
                        score=score, cls_pred=cls_pred)
            valid_instance_msk = instance != self.instance_ignore_label
            pred["void_intersection"] = masks_heatmap[valid_instance_msk].sum(0)
            target = dict(labels=cls_target, masks=masks_target)
            assert (pred['masks_heatmap'] >= 0).all()
            assert (pred['masks_heatmap'] <= 1).all()
            assert (target['masks'] >= 0).all()
            assert (target['masks'] <= 1).all()
            pred, target = self.associate_instances(pred, target)
            # ap
            ap_scores, tgt_idx_to_matched_iou, tgt_idx_to_matched_pred = self.evaluate_matched_ins(pred, target)
            batch_time.update(time.time() - end)

            all_ap = ap_scores["all_ap"]
            all_ap_50 = ap_scores["all_ap_50%"]
            all_ap_25 = ap_scores["all_ap_25%"]
            logger.info('Test initial click: {} [{}/{}]-{} '
                        'Batch time {batch_time.val:.3f} ({batch_time.avg:.3f}) '.format(
                            data_name, data_idx + 1, len(test_loader), segment.size,
                            batch_time=batch_time
                        ))
            logger.info(f"Test result: mAP/AP50/AP25 {all_ap:.4f}/{all_ap_50:.4f}/{all_ap_25:.4f}.")

            all_ap_meter.update(all_ap)
            all_ap_50_meter.update(all_ap_50)
            all_ap_25_meter.update(all_ap_25)
            for (i, class_tag) in enumerate(self.valid_class_tags):
                ap = ap_scores["classes"][class_tag]["ap"]
                ap_50 = ap_scores["classes"][class_tag]["ap50%"]
                ap_25 = ap_scores["classes"][class_tag]["ap25%"]
                ap_class_i = np.zeros(3)            # [ap,ap50,ap25]
                ap_class_i[0] = ap
                ap_class_i[1] = ap_50
                ap_class_i[2] = ap_25
                if np.isnan(np.average(ap_class_i)):
                    # 这个semantic class没出现在本scene中
                    continue
                ap_class_meter_list[i].update(ap_class_i)
            
            # 是否progressive地refine
            if not self.progressive_mode:
                continue
            assert not is_testset, "cannot do progressive mode on test set"
            click_num = 1         # click的个数, 包括了initial click
            # progressive 地 refine
            end = time.time()
            max_click_num = 20
            while click_num < max_click_num:
                click_num += 1
                # 首先更新pred
                tgt_ins_tag_to_matched_iou = {ins_classes[i].item():tgt_idx_to_matched_iou[i] for i in tgt_idx_to_matched_iou.keys()}
                tgt_ins_tag_to_matched_pred = {ins_classes[i].item():tgt_idx_to_matched_pred[i] for i in tgt_idx_to_matched_pred.keys()}
                clicker.update_ious(tgt_ins_tag_to_matched_iou)
                print(tgt_ins_tag_to_matched_iou)
                clicker.update_pred(tgt_ins_tag_to_matched_pred)
                if not clicker.on:
                    # 每个segment类别都满足iou要求, 则停止progressive
                    break
                # merger 继续用 initial click 的, 重复前面的过程
                masks_heatmap = None
                cls_logits = None
                clicks = clicker.make_next_clicks()    # 太费时间, 故每轮对不同fragment统一用一样的clicks
                for i in range(len(fragment_list)):
                    # 对每个切片
                    fragment_batch_size = 1
                    s_i, e_i = i * fragment_batch_size, min((i + 1) * fragment_batch_size, len(fragment_list))
                    pcd_dict = collate_fn(fragment_list[s_i:e_i])         # 取出一个batch中的多个dict，构成一个dict
                    # 转 GPU
                    for key in pcd_dict.keys():
                        if isinstance(pcd_dict[key], torch.Tensor):
                            pcd_dict[key] = pcd_dict[key].cuda(non_blocking=True)
                    
                    idx_part = pcd_dict["index"]      # voxelize产生的，voxel在原数据的point的index
                    pcd_dict['pred_last'] = pred_last_list[i]
                    # clicks = clicker.make_next_clicks(i)
                    input_dict = dict(
                        pcd_dict=pcd_dict, img_dict=img_dict, fusion=fusion, clicker=clicker, clicks=clicks
                    )
                    with torch.no_grad():
                        clicks_dict = model(input_dict)
                        pred_refine = clicks_dict['pred_refine']
                        masks_heatmap_part = pred_refine["masks_heatmap"]
                        cls_logits_part = pred_refine["cls_logits"]
                        pred_last_list[i] = pred_refine
                    if cfg.empty_cache:
                        torch.cuda.empty_cache()
                    # 利用fragment的pred构建成场景的pred
                    masks_heatmap_new = torch.zeros((segment.size, masks_heatmap_part.shape[1])).cuda()
                    # 用三线性插值填充pred到gridsample之前的分辨率
                    masks_heatmap_new[idx_part, :] = masks_heatmap_part
                    idx_part = idx_part.cpu()
                    idx_new = torch.where(~torch.isin(torch.arange(origin_coord.shape[0]), idx_part))[0]
                    coord = origin_coord[idx_part].clone().detach().contiguous()
                    coord_new = origin_coord[idx_new].clone().detach().contiguous()
                    offset = torch.Tensor([coord.shape[0]]).int().cuda()
                    offset_new = torch.Tensor([coord_new.shape[0]]).int().cuda()
                    masks_heatmap_new[idx_new, :] = pointops.interpolation(coord.float(), coord_new.float(), 
                                                                           masks_heatmap_part, offset, offset_new)
                    masks_heatmap_new[masks_heatmap_new<0] = 0
                    cls_logits_new = cls_logits_part
                    if masks_heatmap == None:
                        masks_heatmap = masks_heatmap_new
                        cls_logits = cls_logits_new
                    else:
                        # 融合不同fragment的instance pred
                        masks_heatmap = torch.cat([masks_heatmap, masks_heatmap_new], dim=1)
                        cls_logits = torch.cat([cls_logits, cls_logits_new], dim=0)
                    _, masks_heatmap, cls_logits = merger(masks_heatmap, cls_logits, [])
                
                score, masks_pred, cls_pred, masks_heatmap = self.get_pred(masks_heatmap, cls_logits)
                cls_prob = torch.zeros((cls_pred.shape[0], self.num_classes)).to(masks_heatmap.device)    # [N_pred_ins, num_classes]
                for i in range(cls_pred.shape[0]):
                    ins_tag = cls_pred[i]
                    cls_prob[i, ins_tag] = 1.0
                pred = dict(cls_prob=cls_prob, masks_heatmap=masks_heatmap, masks_pred=masks_pred,
                        score=score, cls_pred=cls_pred)
                pred["void_intersection"] = masks_heatmap[valid_instance_msk].sum(0)
                target = dict(labels=cls_target, masks=masks_target)
                pred, target = self.associate_instances(pred, target)
                ap_scores, tgt_idx_to_matched_iou, tgt_idx_to_matched_pred = self.evaluate_matched_ins(pred, target)
                logger.info(f'Test progressive mode: {data_idx + 1}/{len(test_loader)}-{data_name}, Click: {click_num}/{max_click_num}')

            ins_tag_to_sem_tag = {ins_classes[i]:cls_target[i] for i in range(ins_classes.shape[0])}
            ious_map = clicker.get_ious()       # map: ins_tag -> iou_arr
            nocs_arr = self.compute_noc_metric(ious_map, self.iou_thrs, ins_tag_to_sem_tag)         # len(iou_thrs) x N_tgt_ins
            ious_arr = self.compute_iou_metric(ious_map, self.noc_thrs, ins_tag_to_sem_tag)         # len(noc_thrs) x N_tgt_ins

            number_of_click.update(nocs_arr.sum(-1), n=nocs_arr.shape[-1])      # len(iou_thrs)
            iou_per_click.update(ious_arr.sum(-1), n=ious_arr.shape[-1])        # len(noc_thrs)

            progressive_time.update(time.time() - end)
            logger.info(f'Test progressive mode: {data_name} [{data_idx + 1}/{len(test_loader)}]-{segment.size} '
                        f'Batch {progressive_time.val:.3f} ({progressive_time.avg:.3f}) '
                        f'Number of clicks {self.iou_thrs}: {np.around(number_of_click.avg, 4).tolist()} '
                        f'IoUs {self.noc_thrs}: {np.around(iou_per_click.avg, 4).tolist()} ')
            
        # end for data_idx, data_dict in enumerate(test_loader):
        logger.info("Syncing ...")
        comm.synchronize()
        all_ap_meter_sync = comm.gather(all_ap_meter, dst=0)
        all_ap_50_meter_sync = comm.gather(all_ap_50_meter, dst=0)
        all_ap_25_meter_sync = comm.gather(all_ap_25_meter, dst=0)
        ap_class_meter_sync_list = [comm.gather(ap_class_meter_list[i], dst=0) for i in range(len(self.valid_class_tags))]
        if self.progressive_mode:
            number_of_click_sync = comm.gather(number_of_click, dst=0)
            iou_per_click_sync = comm.gather(iou_per_click, dst=0)

        # 总的test结果
        if comm.is_main_process():
            all_ap = np.sum([meter.sum for meter in all_ap_meter_sync], axis=0) / np.sum([meter.count for meter in all_ap_meter_sync], axis=0)
            all_ap_50 = np.sum([meter.sum for meter in all_ap_50_meter_sync], axis=0) / np.sum([meter.count for meter in all_ap_50_meter_sync], axis=0)
            all_ap_25 = np.sum([meter.sum for meter in all_ap_25_meter_sync], axis=0) / np.sum([meter.count for meter in all_ap_25_meter_sync], axis=0)
            logger.info(f'Val result: mAP/AP50/AP25 {all_ap:.4f}/{all_ap_50:.4f}/{all_ap_25:.4f}.')
            # logger.info('Val result after one click: mAP/AP50/AP25 {all_ap:.4f}/{all_ap_50:.4f}/{all_ap_25:.4f}.')
            if self.progressive_mode:
                noc = np.sum([meter.sum for meter in number_of_click_sync], axis=0) / np.sum([meter.count for meter in number_of_click_sync], axis=0)
                iou = np.sum([meter.sum for meter in iou_per_click_sync], axis=0) / np.sum([meter.count for meter in iou_per_click_sync], axis=0)
                logger.info(f'Val progressive number of click {self.iou_thrs}: {np.around(noc, 4).tolist()}')
                logger.info(f'Val progressive IoU per click {self.noc_thrs}: {np.around(iou, 4).tolist()}')
            for (i, class_tag) in enumerate(self.valid_class_tags):
                ap_class = np.sum([meter.sum for meter in ap_class_meter_sync_list[i]], axis=0) / np.sum([meter.count for meter in ap_class_meter_sync_list[i]], axis=0)
                ap, ap_50, ap_25 = ap_class[0], ap_class[1], ap_class[2]
                logger.info(f'Class_{i}-{cfg.data.names[class_tag]} Result: AP/AP50/AP25 {ap:.4f}/{ap_50:.4f}/{ap_25:.4f}')
                # if self.progressive_mode:
                # noc per class
            logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    @staticmethod
    def collate_fn(batch):
        return batch



# DeprecationWarning

@TESTERS.register_module()
class InteractiveSemSegTester(TesterBase):
    """InteractiveSemSegTester
    for indoor point cloud & RGBD interactive segmentation
    使用 model.forward, 用随机 query
    - refine:
    """
    def __init__(self, pcd_only=False, clicker_cfg=None, random_clicks=False) -> None:
        '''如果pcd_only=True, 就把img_dict和fusion置为None, 并在 .refine 方法中弃用即可'''
        self.pcd_only = pcd_only
        self.clicker_cfg = clicker_cfg
        self.random_clicks = random_clicks          # must be True on test set

    def __call__(self, cfg, test_loader, model):
        assert test_loader.batch_size == 1
        test_dataset = test_loader.dataset
        logger = get_root_logger()
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation Interactive Sem Seg >>>>>>>>>>>>>>>>')

        batch_time = AverageMeter()
        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()
        model.eval()

        save_path = os.path.join(cfg.save_path, "result", "test_epoch{}".format(cfg.test_epoch))
        make_dirs(save_path)
        # create submit folder only on main process
        if "ScanNet" in cfg.dataset_type and comm.is_main_process():
            # in 可以是 string 的包含
            sub_path = os.path.join(save_path, "submit")
            make_dirs(sub_path)
        if 'SemanticKITTI' in cfg.dataset_type and comm.is_main_process():
            sub_path = os.path.join(save_path, "submit")
            make_dirs(sub_path)
        comm.synchronize()
        # fragment inference
        for idx, data_dict in enumerate(test_loader):
            end = time.time()
            # 获取data_dict
            pcd_dict = data_dict[0]['pcd_dict']
            if self.pcd_only:
                img_dict, fusion = None, None
            else:
                # TODO img_dict的获取, 需要修改, 改成后面从click中获取
                img_dict, fusion = test_dataset.get_img_fusion(pcd_dict)
                # 转GPU
                for d in [img_dict, fusion]:
                    for key in d.keys():
                        # collate_fn过了, dict下面一层就是tensor
                        if isinstance(d[key], torch.Tensor):
                            d[key] = d[key].cuda(non_blocking=True)
            # 取出属性
            fragment_list = pcd_dict.pop("fragment_list")  # 切片的点云
            segment = pcd_dict.pop("segment")       # mask
            semantic = torch.Tensor(segment).cuda(non_blocking=True) # 都转到GPU上, 加速运算
            instance = pcd_dict.pop("instance")
            instance = torch.Tensor(instance).cuda(non_blocking=True)
            data_name = pcd_dict.pop("name")       # scene name like 'scene0011_00'
            if not self.pcd_only:
                assert img_dict["scene_id"] == data_name
            pred_save_path = os.path.join(save_path, '{}_pred.pth'.format(data_name))
            # 第一遍refine, TODO 加载保存的结果
            if os.path.isfile(pred_save_path):
                logger.info('{}/{}: {}, loaded pred & refine and label.'.format(idx + 1, len(test_loader), data_name))
                pred = torch.load(pred_save_path).cuda(non_blocking=True)
            else:
                pred = torch.zeros((semantic.shape[0], cfg.data.num_classes)).cuda()     # num_classes 个 mask
                clicker = CLICKER.build(self.clicker_cfg)
                # 如果用 random clicks, 就把真值全设为 ignore label
                clicker.set_state(fragment_list, img_dict=img_dict, fusion=fusion, mode='test',
                                  semantic=semantic if not self.random_clicks else torch.ones_like(semantic) * cfg.semantic_ignore_label, 
                                  instance=instance,
                                  clicks_from_instance=True)
                clicks = None
                for i in range(len(fragment_list)):
                    # 对每个切片(不同切片，意味着对一个voxel内多个point的，会遍历point的颜色作为voxel的颜色)
                    fragment_batch_size = 1
                    s_i, e_i = i * fragment_batch_size, min((i + 1) * fragment_batch_size, len(fragment_list))
                    pcd_dict = collate_fn(fragment_list[s_i:e_i])         # 取出一个batch中的多个dict，构成一个dict
                    # 转 GPU
                    for key in pcd_dict.keys():
                        if isinstance(pcd_dict[key], torch.Tensor):
                            pcd_dict[key] = pcd_dict[key].cuda(non_blocking=True)
                    
                    idx_part = pcd_dict["index"]      # voxelize产生的，voxel在原数据的point的index
                    pcd_dict["fragment_idx"] = i
                    # print(pcd_dict['discrete_coord'].shape[0])
                    input_dict = dict(
                        pcd_dict=pcd_dict, img_dict=img_dict, fusion=fusion, clicker=clicker
                    )
                    # 由于discrete_coords一致(实为fps太费时间), 因此后续click坐标全部沿用 fragment_list[0] 产生的clicks
                    if self.random_clicks and clicks != None:
                        input_dict["clicks"] = clicks

                    with torch.no_grad():
                        # interactive 的部分
                        res = model(input_dict)
                        pred_part = res["seg_logits"]
                        clicks = res["clicks_dict"]["clicks"]
                        pred_part = F.softmax(pred_part, -1)      # 为什么再做一遍softmax？因为输出的是并未归一化的, 俗称logits
                    if cfg.empty_cache:
                        torch.cuda.empty_cache()
                    bs = 0
                    for be in pcd_dict["offset"]:
                        # 映射回原数据的shape，其实这个for循环只做了一次，因为在Collect类里只产生一个offset
                        pred[idx_part[bs: be], :] += pred_part[bs: be]
                        bs = be
                    logger.info('Test initial click: {}/{}-{data_name}, Batch: {batch_idx}/{batch_num}'.format(
                        idx + 1, len(test_loader), data_name=data_name, batch_idx=i, batch_num=len(fragment_list)))
                # 这里对多个切片的预测的概率，相加得到pred（因为不同part之间会有重叠的point）
                # 取概率最大的类别，作为预测的类别
                pred = pred.max(1)[1]
                # 保存结果
                torch.save(pred, pred_save_path)
            
            # 计算并更新meter
            intersection, union, target = intersection_and_union_gpu(pred, semantic, cfg.data.num_classes,
                                                                 cfg.data.ignore_index)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection)
            union_meter.update(union)
            target_meter.update(target)

            # 计算这个scene的miou
            mask = union != 0
            iou_class = intersection / (union + 1e-10)
            iou = np.mean(iou_class[mask])                  # 为什么要mask？因为是每个类别都计算，而某些类别并不会在这个scene中出现
            acc = sum(intersection) / (sum(target) + 1e-10)
            m_iou = np.mean(intersection_meter.sum / (union_meter.sum + 1e-10))         # 这里的m_iou不是按class平均
            m_acc = np.mean(intersection_meter.sum / (target_meter.sum + 1e-10))

            batch_time.update(time.time() - end)
            logger.info('Test initial click: {} [{}/{}]-{} '
                        'Batch time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Accuracy {acc:.4f} ({m_acc:.4f}) '
                        'mIoU {iou:.4f} ({m_iou:.4f}) '.format(
                            data_name, idx + 1, len(test_loader), semantic.shape[0],
                            batch_time=batch_time, 
                            acc=acc, m_acc=m_acc,
                            iou=iou, m_iou=m_iou
                        ))
            # submit
            if "ScanNet" in cfg.dataset_type:
                np.savetxt(os.path.join(save_path, "submit", '{}.txt'.format(data_name)),
                           test_dataset.class2id[pred.cpu().numpy()].reshape([-1, 1]), fmt="%d")
            if "SemanticKITTI" in cfg.dataset_type:
                # 00_000000 -> 00, 000000
                sequence_name, frame_name = data_name.split("_")
                os.makedirs(
                    os.path.join(save_path, "submit", "sequences", sequence_name, "predictions"), exist_ok=True
                )
                pred = pred.astype(np.uint32)
                pred = np.vectorize(cfg.learning_map_inv.__getitem__)(pred).astype(np.uint32)
                pred.tofile(
                    os.path.join(save_path, "submit", "sequences", sequence_name, "predictions", f"{frame_name}.label")
                )
            
        logger.info("Syncing ...")
        comm.synchronize()
        intersection_meter_sync = comm.gather(intersection_meter, dst=0)
        union_meter_sync = comm.gather(union_meter, dst=0)
        target_meter_sync = comm.gather(target_meter, dst=0)

        # 总的test结果
        if comm.is_main_process():
            intersection = np.sum([meter.sum for meter in intersection_meter_sync], axis=0)
            union = np.sum([meter.sum for meter in union_meter_sync], axis=0)
            target = np.sum([meter.sum for meter in target_meter_sync], axis=0)

            iou_class = intersection / (union + 1e-10)
            accuracy_class = intersection / (target + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(intersection) / (sum(target) + 1e-10)

            logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}'.format(mIoU, mAcc, allAcc))
            for i in range(cfg.data.num_classes):
                logger.info('Class_{idx} - {name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}'.format(
                                idx=i, name=cfg.data.names[i], 
                                iou=iou_class[i], accuracy=accuracy_class[i]
                            ))
            logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    @staticmethod
    def collate_fn(batch):
        return batch