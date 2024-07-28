
1. pcd dataset
    - 要求 test dataset dict 的 keys 为 
        - fragment_list = pcd_dict.pop("fragment_list")  # 切片的点云
            - fragment.keys(): ['coord', 'discrete_coord', 'index', 'color', 'scene_id', 'offset', 'feat']
        - segment = pcd_dict.pop("segment")      # mask
        - data_name = pcd_dict.pop("name")       # scene name like 'scene0011_00'
    - 在 prepare test data 的时候，直接.pop("segment")了，省内存；后续gridsample的时候的keys里就不能有segment了，直接用index能索引得到各个fragment的segment
    - 要求 batched train dataset dict 的keys为
        - ['coord', 'discrete_coord', 'segment', 'color', 'scene_id', 'offset', 'feat']
        - len(dict["offset"]) > 1
2. 软链接
    - `ln -s ${PROCESSED_DATASET_DIR} ${CODEBASE_DIR}/data/dataset_name`
    - `/data/shared_workspace/yuzijian/datasets/*`，`/data/shared_workspace/yuzijian/ws/ISeg3D/data/*`
    - 填绝对路径