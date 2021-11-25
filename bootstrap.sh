#!/usr/bin/bash
set -e
wget "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh" -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
~/miniconda/bin/conda init $(echo $SHELL | awk -F '/' '{print $NF}')
echo 'Successfully installed miniconda...'
echo -n 'Conda version: '
~/miniconda/bin/conda --version
echo -e '\n'
exec bash
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
# conda env reate -f environment.yaml
conda create --name tf2 python=3.7 
conda activate tf2  
conda install -c anaconda seaborn
conda install matplotlib
pip install --upgrade tensorflow
pip install wordcloud
