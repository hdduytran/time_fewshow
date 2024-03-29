#!/bin/bash
#bash scripts/train_proto.sh 0 2 linear_transform ./exp/train.py

if [ "$#" -ne 5 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for the GPUs, n_samples, arch "
  exit 1 
fi
gpus=$1
lr=2e-3
batch_size=512
weight_decay=1e-5
num_support_tr=$2
num_query_tr=$2
num_support_val=$2
num_query_val=$2
arch=$3
pythonfile=$4
model_name=$3
dataset_root=$5

CUDA_VISIBLE_DEVICES=${gpus} python ${pythonfile} \
	       --arch ${arch} --dataset_root ${dataset_root} --workers 2 \
	       --log_dir ./logs/5shot_${arch}/ --log_interval 100 --test_interval 100 \
	       --epochs 1000 --iterations 100 --batch_size ${batch_size} --lr ${lr} --lr_step 10000 --lr_gamma 0.7 --weight_decay ${weight_decay} \
	       --num_support_tr ${num_support_tr} --num_query_tr ${num_query_tr} --num_support_val ${num_support_val}  --num_query_val ${num_query_val} \
         --model_name ${model_name}


