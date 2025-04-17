#!/bin/bash

set -e

source venv/bin/activate

DEFAULT_CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES="${1:-$DEFAULT_CUDA_VISIBLE_DEVICES}"

export CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
NCCL_P2P_DISABLE=1
NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE
export NCCL_IB_DISABLE


model_name='./mymodels/gpt2_numeric'

datasets=(
discrete14_3_05_lrelu_l2_s00
discrete14_3_05_lrelu_l2_s01
discrete14_3_05_lrelu_l2_s02
discrete14_3_05_lrelu_l2_s03
discrete14_3_05_lrelu_l2_s04
discrete14_3_10_lrelu_l2_s00
discrete14_3_10_lrelu_l2_s01
discrete14_3_10_lrelu_l2_s02
discrete14_3_10_lrelu_l2_s03
discrete14_3_10_lrelu_l2_s04
discrete14_2_10_lrelu_l2_s00
discrete14_2_10_lrelu_l2_s01
discrete14_2_10_lrelu_l2_s02
discrete14_2_10_lrelu_l2_s03
discrete14_2_10_lrelu_l2_s04
discrete14_3_10_lrelu_l2_s00
discrete14_3_10_lrelu_l2_s01
discrete14_3_10_lrelu_l2_s02
discrete14_3_10_lrelu_l2_s03
discrete14_3_10_lrelu_l2_s04
discrete14_4_10_lrelu_l2_s00
discrete14_4_10_lrelu_l2_s01
discrete14_4_10_lrelu_l2_s02
discrete14_4_10_lrelu_l2_s03
discrete14_4_10_lrelu_l2_s04
discrete14_5_10_lrelu_l2_s00
discrete14_5_10_lrelu_l2_s01
discrete14_5_10_lrelu_l2_s02
discrete14_5_10_lrelu_l2_s03
discrete14_5_10_lrelu_l2_s04
)

for dataset in ${datasets[@]};do
    python3 trainer.py  \
            --model_name=${model_name}  \
            --dataset_dir=./data/${dataset}  \
            --batch_size=384 \
            --logging_step=2000 \
            --output_name=${dataset}
done

