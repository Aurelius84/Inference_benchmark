#!/bin/bash

current_dir=$(cd "$(dirname "$0")";pwd)

# resnet50、resnet101
python $current_dir/resnet.py
