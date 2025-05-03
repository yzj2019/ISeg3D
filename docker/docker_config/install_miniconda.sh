#!/bin/sh
mkdir -p ~/miniconda3
# curl -fSL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -o ~/miniconda3/miniconda.sh
# bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
# rm -rf ~/miniconda3/miniconda.sh
# 网络不好时，使用本地miniconda.sh
bash /tmp/resources/miniconda.sh -b -u -p ~/miniconda3

# 初始化conda
~/miniconda3/bin/conda init bash

# 换base环境里的pip源
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple/