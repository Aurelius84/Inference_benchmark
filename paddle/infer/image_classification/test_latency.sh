#!/bin/bash

repeat_time=100
current_dir=$(cd "$(dirname "$0")";pwd)
model_dir= $current_dir/../../../models

# save logs
mkdir -p $current_dir/logs
# step1. build executable file
bash $current_dir/re_build.sh

# test resnet
for model_name in "resnet50" "resnet101" "mobilenet_v1"
do
    for model_type in "static" "dy2stat"
    do
        for device in "use_gpu" "nouse_gpu"
        do
            for batch_size in 1 4 16 32
            do
                model_path=$model_dir/$model_name/$model_type
                log_path=$current_dir/logs/resnet_${model_type}_${device}_$batch_size.txt
                touch $log_path
                # test latency
                ./image_classification_exe --$device --batch_size=$batch_size --repeat_time=$repeat_time --dirname=$model_path > $log_path 2>&1
                wait
            done
        done
    done
done
