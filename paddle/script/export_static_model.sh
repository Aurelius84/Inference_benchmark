#!/bin/bash

current_dir=$(cd "$(dirname "$0")";pwd)

model_files=(
    "resnet" 
    "mobilenet_v1"
    )

for model in ${model_files[@]}
do
  echo "saving $model ..."
  python $current_dir/$model.py
  wait
done
