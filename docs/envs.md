# 环境配置

## 1. example

```bash
conda create --name iseg3d python=3.8.16 -y
conda activate iseg3d
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 1. pytorch
# Choose version you want here: https://pytorch.org/get-started/previous-versions/ , with nvcc -V
# cuda >= 11.3, pytorch == 1.11.0 or 1.12.0
conda install -n iseg3d pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y
# PTv3, cuda >= 11.7, pytorch >= 1.12.0
# conda install -n iseg3d pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
# conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 2. pointcept common
conda install -n iseg3d ninja -y
conda install -n iseg3d h5py pyyaml -c anaconda -y
conda install -n iseg3d sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install -n iseg3d pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric easydict
pip install open3d pyviz3d opencv-python

# 3. spconv (for SparseUNet)
# version must be satisfied with cudatoolkit
# refer https://github.com/traveller59/spconv
pip install spconv-cu113
# for PTv3
# pip install spconv-cu118
# pip install spconv-cu120

# 4. Flash attention
MAX_JOBS=4 pip install flash-attn==0.2.2 --no-build-isolation
# PTv3, latest flash-attn v2, need >= cu117
# MAX_JOBS=4 pip install flash-attn --no-build-isolation

# 5. libs
# 5.1. PTv1 & PTv2 or precise eval or mask 3d
cd libs/pointops
# usual
# python setup.py install
# docker & multi GPU arch
TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6" python  setup.py install
# e.g. 7.5: RTX 3000; 8.0: A100; 8.6: 3090
# You should modify this to match your GPU compute capability, see https://developer.nvidia.cn/zh-cn/cuda-gpus#compute
cd ../..
# 5.2. pointgroup
conda install -n iseg3d -c bioconda google-sparsehash -y
cd libs/pointgroup_ops
python setup.py install --include_dirs=${CONDA_PREFIX}/include
cd ../..
# 5.3. scannet segmentator
conda install -n iseg3d numba -y
cd libs/scannet_segmentator
pip install .
cd ../..
# 5.4. superpoint graph
cd libs/superpoint_graph
pip install git+https://github.com/drprojects/point_geometric_features.git
conda install -n iseg3d omegaconf numpy pyyaml -y
pip install .
cd ../..
# 5.5. interactive visualization
# TBD

# 6. optional
# # sam-hq
# pip install opencv-python pycocotools matplotlib onnxruntime onnx
# # git clone
# cd ../sam-hq
# pip install -e .
# cd ../iseg3d
# # sam3d
# pip install scikit-image imageio argparse
```

## 2. PTv3

```bash
conda create --name iseg3d_ptv3 python=3.8.16 -y
conda activate iseg3d_ptv3
# pip 换源
# pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 1. pytorch
# conda install -n iseg3d_ptv3 pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
# conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y


# 2. pointcept common
conda install -n iseg3d_ptv3 ninja -y
conda install -n iseg3d_ptv3 h5py pyyaml -c anaconda -y
conda install -n iseg3d_ptv3 sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm ipykernel -c conda-forge -y
conda install -n iseg3d_ptv3 pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric easydict opencv-python
pip install open3d pyviz3d

# 3. spconv (for SparseUNet)
# pip install spconv-cu118
# pip install spconv-cu117
pip install spconv-cu120

# 4. Flash attention
MAX_JOBS=4 pip install flash-attn --no-build-isolation

# 5. libs
# 5.1. PTv1 & PTv2 or precise eval or mask 3d
cd libs/pointops
# usual
# python setup.py install
# docker & multi GPU arch
TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6" python  setup.py install
# e.g. 7.5: RTX 3000; 8.0: A100; 8.6: 3090
# You should modify this to match your GPU compute capability, see https://developer.nvidia.cn/zh-cn/cuda-gpus#compute
cd ../..
# 5.2. pointgroup
conda install -n iseg3d_ptv3 -c bioconda google-sparsehash -y
cd libs/pointgroup_ops
python setup.py install --include_dirs=${CONDA_PREFIX}/include
cd ../..

cd ../sam-hq
pip install -e .
cd ../ISeg3D

```

## A. Flash Attention Installation

`undefined symbol` 错误，通常是版本不匹配。

寻找对应的版本：去到 github 库 的 [publish.yml](https://github.com/Dao-AILab/flash-attention/blob/v2.3.0/.github/workflows/publish.yml) 文件，找到对应的版本号。
```yml
torch-version: ['1.12.1', '1.13.1', '2.0.1', '2.1.0.dev20230731']
cuda-version: ['11.6.2', '11.7.1', '11.8.0', '12.1.0', '12.2.0']
```

查看排除项，确定哪些组合不受支持：
```yml
exclude:
    # Pytorch <= 1.12 does not support Python 3.11
    - torch-version: '1.12.1'
      python-version: '3.11'
    ...
```