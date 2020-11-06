#!/bin/bash

model_type=$1
batch_size=$2
dirname=$3

for i in {1..5}
do
  echo "start iter : $i ..."
  ./$model_type/${model_type}_exe --use_gpu --batch_size=$batch_size --repeat_time=100 --dirname=$dirname &
  wait
done
