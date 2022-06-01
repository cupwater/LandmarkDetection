#!/bin/bash

python -m torch.distributed.launch \
   --nproc_per_node=4 \
   train_fp16.py --config-file $1
