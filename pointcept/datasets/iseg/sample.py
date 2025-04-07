import os
import argparse
import torch
import numpy as np
import tqdm
from functools import partial
import sys

sys.path.append(".")
from pointcept.engines.defaults import (
    default_config_parser,
    default_setup,
)
from pointcept.datasets import build_dataset
import pointops


def sample(
    dataset_cfg,
    sample_type,
    sample_num,
    sample_func,
    semantic_ignore=-1,
    instance_ignore=-1,
):
    dataset = build_dataset(dataset_cfg)
    dataset.loop = 1
    for idx in tqdm.trange(len(dataset)):
        # sample dataset raw data
        data_dict = dataset.get_data(idx)
        save_dir = dataset.data_list[idx]
        sem_ignore_mask = data_dict["segment"] != semantic_ignore
        ins_ignore_mask = data_dict["instance"] != instance_ignore
        ignore_mask = sem_ignore_mask & ins_ignore_mask  # 确保采样的点都是关心点
        sampled_idx = sample_func(data_dict, sample_num, ignore_mask)
        np.save(os.path.join(save_dir, f"sampled_idx_{sample_type}.npy"), sampled_idx)


def fps_sampling(data_dict, sample_num, ignore_mask):
    inverse = torch.where(ignore_mask)[0]  # 换原索引
    p = (
        torch.tensor(data_dict["coord"][ignore_mask], dtype=torch.float32)
        .cuda()
        .contiguous()
    )
    o = torch.tensor([p.shape[0]], dtype=torch.long).cuda()
    n_o = torch.tensor([sample_num], dtype=torch.long).cuda()
    with torch.no_grad():
        sampled_idx = pointops.farthest_point_sampling(p, o, n_o)
        sampled_idx = inverse[sampled_idx]
    return sampled_idx.cpu().numpy()


def random_sampling(data_dict, sample_num, ignore_mask):
    inverse = torch.where(ignore_mask)[0]  # 换原索引
    p = torch.tensor(data_dict["coord"][ignore_mask], dtype=torch.float32).contiguous()
    sampled_idx = torch.randperm(p.shape[0], device="cpu")[:sample_num]
    sampled_idx = inverse[sampled_idx]
    return sampled_idx.cpu().numpy()


SAMPLE_TYPE = {
    "fps": {
        "func": fps_sampling,
        "name": "farthest point",
    },
    "rand": {
        "func": random_sampling,
        "name": "random",
    },
}

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--sample_type", type=str, default="fps", help="采样方式")
    parser.add_argument(
        "-d", "--dataset_name", type=str, default="scannet", help="数据集名称"
    )
    parser.add_argument(
        "-c", "--cfg_name", type=str, default="insseg-mask3d-spunet-0", help="配置文件名称"
    )
    parser.add_argument("-n", "--sample_num", type=int, default=200, help="采样点数量")
    args = parser.parse_args()

    # 使用命令行参数
    sample_type = args.sample_type
    dataset_name = args.dataset_name
    cfg_name = args.cfg_name
    sample_num = args.sample_num

    assert sample_type in [
        "fps",
        "rand",
        "sem",
        "ins",
        "super",
    ], f"unsupported sample type {sample_type}"
    assert dataset_name in [
        "scannet",
        "s3dis",
        "kitti360",
    ], f"unsupported dataset {dataset_name}"

    sample_name = SAMPLE_TYPE[sample_type]["name"]
    sample_func = SAMPLE_TYPE[sample_type]["func"]
    config_file = f"configs/{dataset_name}/{cfg_name}.py"
    cfg = default_config_parser(config_file, None)
    cfg = default_setup(cfg)
    print(
        f"start {sample_name} sampling for {dataset_name} dataset into {sample_num} points"
    )
    print("sample train dataset")
    sample(cfg.data.train, sample_type, sample_num, sample_func)
    print("sample val dataset")
    sample(cfg.data.val, sample_type, sample_num, sample_func)
    print("sample test dataset")
    cfg.data.test.split = "test"
    sample(cfg.data.test, sample_type, sample_num, sample_func)
    print("done")
