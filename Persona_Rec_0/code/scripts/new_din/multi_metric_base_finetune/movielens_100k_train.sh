export CUDA_VISIBLE_DEVICES=3
#cd ../../..
#rm model.tar.gz
#tar -czf model.tar.gz ./main_new ./loader ./model ./util ./module
#cd scripts/new_DIN/base

#ITERATION=30000
ITERATION=80000
#SNAPSHOT=10000
SNAPSHOT=8000
#ARCH_CONF_FILE=`pwd`/movielens_conf.json
#ARCH_CONFIG_CONTENT=`cat ${ARCH_CONF_FILE} | base64 -w 0`
ARCH_CONF_FILE="../movielens_100k_conf.json"


GRADIENT_CLIP=5                     # !
BATCH_SIZE=1024

######################################################################
LEARNING_RATE=0.0001
CHECKPOINT_PATH=../../../checkpoint/SIGIR2023/movielens_100k_din/
######################################################################

echo ${CHECKPOINT_PATH}
echo "Model save to ${CHECKPOINT_PATH}"


USER_DEFINED_ARGS="--model=din --num_loading_workers=1 --arch_config=${ARCH_CONFIG_CONTENT} --learning_rate=${LEARNING_RATE} \
--max_gradient_norm=${GRADIENT_CLIP} --batch_size=${BATCH_SIZE} --snapshot=${SNAPSHOT} --max_steps=${ITERATION} --checkpoint_dir=${CHECKPOIN\
T_PATH} --arch_config=${ARCH_CONF_FILE}"

dataset="../../../../data/ml-100k/ood_generate_dataset_tiny_10_30u30i"
#train_file="${dataset}/train.txt"
#test_file="${dataset}/test.txt"
#data="${train_file},${test_file}"

echo ${USER_DEFINED_ARGS}
python ../../../main_new/multi_metric_vanilla_train2_finetune.py \
--dataset=${dataset} \
${USER_DEFINED_ARGS}

echo "Training done: ${CHECKPOINT_PATH}"
