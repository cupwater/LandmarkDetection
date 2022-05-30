<!--
 * @Author: Peng Bo
 * @Date: 2022-05-21 22:53:26
 * @LastEditTime: 2022-05-21 22:55:53
 * @Description: 
 * 
-->
## Landmarks Detection for Chest Image

### Usage
1. training on single gpu:
```bash
 python train.py --config-file your_config_file
```
e.g. ```python train.py --config-file experiments/template/landmark_detection_template.yaml```


2. training on multiple gpus (nodes):
```bash
python -m torch.distributed.launch \
   --nproc_per_node=4 \
   train_fp16.py --config-file your_config_file
```
