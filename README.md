
## 1. 环境配置

```bash
conda create --name iseg3d python=3.8.16 -y
conda activate iseg3d
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# Choose version you want here: https://pytorch.org/get-started/previous-versions/ , with nvcc -V
# cuda >= 11.3 , pytorch == 1.11.0 or 1.12.0
conda install -n iseg3d pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y

# pointcept
conda install -n iseg3d ninja -y
conda install -n iseg3d h5py pyyaml -c anaconda -y
conda install -n iseg3d sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install -n iseg3d pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
conda deactivate
conda activate iseg3d      # 老是有掉conda环境的bug
pip install torch-geometric easydict
# spconv (SparseUNet), version must be satisfied with cudatoolkit
# refer https://github.com/traveller59/spconv
pip install spconv-cu113
# PTv1 & PTv2 or precise eval or mask 3d
cd libs/pointops
# usual
# python setup.py install
# docker & multi GPU arch
TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6" python  setup.py install
# TORCH_CUDA_ARCH_LIST="7.5;8.0" python  setup.py install
# e.g. 7.5: RTX 3000; 8.0: a100; 8.6: 3090
# You should modify this to match your GPU compute capability, see https://developer.nvidia.cn/zh-cn/cuda-gpus#compute
cd ../..
# pointgroup
conda install -n iseg3d -c bioconda google-sparsehash -y
cd libs/pointgroup_ops
python setup.py install --include_dirs=${CONDA_PREFIX}/include
cd ../..
# scannet segmentator
conda install -n iseg3d numba -y
cd libs/scannet_segmentator
pip install .
cd ../..
# superpoint graph
cd libs/superpoint_graph
pip install git+https://github.com/drprojects/point_geometric_features.git
conda install -n iseg3d omegaconf numpy pyyaml -y
pip install .
cd ../..

# visualization, optional
pip install open3d PyQt5

# sam-hq
pip install opencv-python pycocotools matplotlib onnxruntime onnx
# git clone
cd ../sam-hq
pip install -e .
cd ../iseg3d
# sam3d
pip install scikit-image imageio argparse

# Flash attention
MAX_JOBS=4 pip install flash-attn==0.2.2 --no-build-isolation
```

## 2. 数据准备&预处理

### 2.1. Scannetv2

详细预处理流程见`dataset/scannetv2/preprocess/README.md`。

#### 2.1.1. download & unzip

使用[opendatalab](https://opendatalab.com/)的api进行下载，会比官方快很多：
```bash
pip install opendatalab
odl login -u username -p password
odl get    ScanNet_v2
```

分卷合并，解压：
```bash
cd ScanNet_v2
mkdir unzip
cd raw
cat scans.tar.* > scans.tar
tar -xvf scans.tar -C ../unzip
tar -xvf scans_test.tar -C ../unzip
```

解压后得到`unzip/scans/`和`unzip/scans_test`两个文件夹。

#### 2.1.2. 2d images 

在`dataset/scannetv2/preprocess/`文件夹下，activate虚拟环境并执行`nohup bash ./prepare_2d_data.sh > nohup.log 2>&1 &`即可，关于路径等参数在脚本里自行修改。

#### 2.1.3. 3d pointclouds

`python preprocess_3d_data.py --dataset_root /data/shared_workspace/yuzijian/datasets/ScanNet_v2/unzip --output_root /data/shared_workspace/yuzijian/ws/iseg3d/dataset/scannetv2/scannetv2_pointclouds`

### 2.2. S3DIS

