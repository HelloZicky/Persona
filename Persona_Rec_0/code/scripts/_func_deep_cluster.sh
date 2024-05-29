#cd ../../..
#rm model.tar.gz
#tar -czf model.tar.gz ./main_new ./loader ./model ./util ./module
#cd scripts/new_DIN/base

shell_name=$0
dataset=$1
model=$2
class_num=$3
line_num=$4
cuda_num=$5

#ITERATION=30000
ITERATION=3600
#SNAPSHOT=10000
SNAPSHOT=360
#ARCH_CONF_FILE=`pwd`/amazon_conf.json
#ARCH_CONFIG_CONTENT=`cat ${ARCH_CONF_FILE} | base64 -w 0`
ARCH_CONF_FILE="new_${model}/${dataset}_conf.json"


GRADIENT_CLIP=5                     # !
#BATCH_SIZE=1024
BATCH_SIZE=512

######################################################################
LEARNING_RATE=0.001
CHECKPOINT_PATH=../checkpoint/SIGIR2023/${dataset}_${model}/
######################################################################

echo ${CHECKPOINT_PATH}
echo "Model save to ${CHECKPOINT_PATH}"


USER_DEFINED_ARGS="--model=deep_cluster --num_loading_workers=1 --arch_config=${ARCH_CONFIG_CONTENT} --learning_rate=${LEARNING_RATE} \
--max_gradient_norm=${GRADIENT_CLIP} --batch_size=${BATCH_SIZE} --snapshot=${SNAPSHOT} --max_steps=${ITERATION} --checkpoint_dir=${CHECKPOIN\
T_PATH} --arch_config=${ARCH_CONF_FILE} --class_num=${class_num} --base_model=${model}"

#dataset="../../../../data/Amazon_cds/ood_generate_dataset_tiny"
#dataset="../../../../data/Amazon_cds/ood_generate_dataset_tiny_10_30u30i"
#line_num={$1}
#dataset="../../../../data/Amazon_cds/ood_generate_dataset_large"
#train_file="${dataset}/train.txt"
#test_file="${dataset}/test.txt"
#data="${train_file},${test_file}"
train_file="../fig_${dataset}_kmeans/amazon_${line_num}_${dataset}_${model}_classnum${class_num}.txt"
test_folder="../checkpoint/SIGIR2023/${dataset}_${model}/base"
test_file="${test_folder}/train_hist_embed.txt,${test_folder}/test_hist_embed.txt"
data="${train_file};${test_file}"

echo ${USER_DEFINED_ARGS}
export CUDA_VISIBLE_DEVICES=${cuda_num}
python ../main_new/deep_cluster.py \
--dataset=${data} \
${USER_DEFINED_ARGS}

echo "Training done: ${CHECKPOINT_PATH}"

