#!/bin/bash

repeat_time=100
current_dir=$(cd "$(dirname "$0")";pwd)
echo $current_dir
model_dir=$(cd "$(dirname "$0")";cd ../../models;pwd)
echo $model_dir

# save logs
mkdir -p $current_dir/logs
# step1. build executable file
bash $current_dir/re_build.sh

# test resnet
for model_name in "resnet50"
do
    for model_type in "static" "dy2stat"
    do   
        # For now just test GPU latency(defalut gpu:0), speicific "nouse_gpu" while on CPU
        for device in "use_gpu"
        do
            for batch_size in 1 4 16 32
            do
                model_path=$model_dir/$model_type/$model_name
                log_path=$current_dir/logs/resnet_${model_type}_${device}_$batch_size.txt
                touch $log_path
                # test latency
                ./image_classification_exe --$device --batch_size=$batch_size --repeat_time=$repeat_time --dirname=$model_path > $log_path 2>&1
                wait
            done
        done
    done
done
