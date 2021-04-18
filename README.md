[WIP] Early reimplementation of https://charigyang.github.io/motiongroup/

# Usage
```shell
# download DAVIS

mkdir -p data/common
wget -nc -P data/common \
  https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip \
  https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip \
  https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-challenge-480p.zip
unzip -d data/common -o 'data/common/DAVIS-2017-*-480p.zip'

# generate flow
python gen_raft_flow.py
```
