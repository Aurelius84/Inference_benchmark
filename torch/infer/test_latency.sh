#!/bin/bash

model_type=$1
dirname=$2

for i in {1..5}
do 
  echo "start run iter : $i ..."
  ./${model_type}_exe $dirname &
  wait
done
