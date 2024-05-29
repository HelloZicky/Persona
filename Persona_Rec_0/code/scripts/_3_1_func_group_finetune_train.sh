#export CUDA_VISIBLE_DEVICES=3
#cd ../../..
#rm model.tar.gz
#tar -czf model.tar.gz ./main_new ./loader ./model ./util ./module
#cd scripts/new_din/base

#shell_name=$0
#dataset=$1
#model=$2
#class_num=$3
#line_num=$4
#cuda_num=$5

dataset=$1
model=$2
group_num=$3
cuda_num=$4

#ITERATION=30000
ITERATION=24300
#SNAPSHOT=10000
SNAPSHOT=2430
MAX_EPOCH=20
#ARCH_CONF_FILE=`pwd`/amazon_conf.json
#ARCH_CONFIG_CONTENT=`cat ${ARCH_CONF_FILE} | base64 -w 0`
#ARCH_CONF_FILE="../amazon_cds_conf.json"
ARCH_CONF_FILE="new_${model}/${dataset}_conf.json"


GRADIENT_CLIP=5                     # !
BATCH_SIZE=1024

######################################################################
LEARNING_RATE=0.001
#CHECKPOINT_PATH=../../../checkpoint/SIGIR2023/amazon_cds_din/meta
#CHECKPOINT_PATH=../checkpoint/SIGIR2023/${dataset}_${model}/meta_grad_group_${group_num}
CHECKPOINT_PATH=../checkpoint/NIPS2023/${dataset}_${model}/meta_grad_group_${group_num}
#pretrain_model_path=../checkpoint/SIGIR2023/${dataset}_${model}/base/best_auc.pkl
#pretrain_model_path=../checkpoint/SIGIR2023/${dataset}_${model}/2_prototype_long_tail/best_auc_${group_num}.pkl
#pretrain_model_path=../checkpoint/NIPS2023/${dataset}_${model}/2_prototype_long_tail/best_auc_${group_num}.pkl
pretrain_model_path=../checkpoint/NIPS2023/${dataset}_${model}/meta_grad_gru/best_auc.pkl
######################################################################

echo ${CHECKPOINT_PATH}
echo "Model save to ${CHECKPOINT_PATH}"


USER_DEFINED_ARGS="--model=meta_${model}_prototype_grad --num_loading_workers=1 --arch_config=${ARCH_CONFIG_CONTENT} --learning_rate=${LEARNING_RATE} \
--max_gradient_norm=${GRADIENT_CLIP} --batch_size=${BATCH_SIZE} --snapshot=${SNAPSHOT} --max_steps=${ITERATION} --checkpoint_dir=${CHECKPOIN\
T_PATH} --arch_config=${ARCH_CONF_FILE} --pretrain_model_path=${pretrain_model_path} --pretrain --base_model=${model} --group_num=${group_num}"

#dataset="../../../../data/Amazon_cds/generate_dataset"
#dataset="../../../../data/Amazon_cds/ood_generate_dataset_tiny_10_30u30i"
dataset="../../data/${dataset^}/ood_generate_dataset_tiny_10_30u30i"

#train_file="${dataset}/train.txt"
#test_file="${dataset}/test.txt"
#data="${train_file},${test_file}"
data=${dataset}

export CUDA_VISIBLE_DEVICES=${cuda_num}
echo ${USER_DEFINED_ARGS}
#python ../../../main_new/multi_metric_meta_train2.py \
#python ../main_new/multi_metric_meta_grad_train2.py \
python ../main_new/multi_metric_meta_grad_group_train2.py \
--dataset=${data} \
${USER_DEFINED_ARGS}

echo "Training done: ${CHECKPOINT_PATH}"

