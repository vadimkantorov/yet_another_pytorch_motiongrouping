[WIP] Early reimplementation of https://charigyang.github.io/motiongroup/ (https://arxiv.org/abs/2104.07658)

# Usage
```shell
mkdir -p data/common
git clone https://github.com/davisvideochallenge/davis2017-evaluation data/common/davis2017-evaluation

# download DAVIS
wget -nc -P data/common \
  https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip \
  https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip \
  https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-challenge-480p.zip
unzip -d data/common -o 'data/common/DAVIS-2017-*-480p.zip'

# download RAFT checkpoints 
wget -nc -P data/common \
  https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip \
  https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip \
  https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-challenge-480p.zip
unzip -d data/common -o 'data/common/DAVIS-2017-*-480p.zip'

# generate optical flow on CUDA device 0
python gen_raft_flow.py --device cuda:0 --dataset-split-name val   --dataset-root-dir data/common/DAVIS --dataset-root-dir-flow data/common/DAVISflow
python gen_raft_flow.py --device cuda:0 --dataset-split-name train --dataset-root-dir data/common/DAVIS --dataset-root-dir-flow data/common/DAVISflow

# evaluate on DAVIS-2016
python davis.py --root data/common/DAVIS -i data/saved -o data/converted
( cd data/common/davis2017-evaluation && python evaluation_method.py  --set val --task semi-supervised --davis_path ../DAVIS --results_path ../../converted )
```
