export CUDA_VISIBLE_DEVICES=1
#cd ../../..
#rm model.tar.gz
#tar -czf model.tar.gz ./main_new ./loader ./model ./util ./module
#cd scripts/new_gru4rec/base

#ITERATION=30000
ITERATION=94400
#SNAPSHOT=10000
SNAPSHOT=9440
MAX_EPOCH=20
#ARCH_CONF_FILE=`pwd`/amazon_conf.json
#ARCH_CONFIG_CONTENT=`cat ${ARCH_CONF_FILE} | base64 -w 0`
ARCH_CONF_FILE="../amazon_cds_conf.json"


GRADIENT_CLIP=5                     # !
BATCH_SIZE=1024

######################################################################
LEARNING_RATE=0.001
CHECKPOINT_PATH=../../../checkpoint/SIGIR2023/amazon_cds_gru4rec/meta_ood_if_apg
######################################################################

echo ${CHECKPOINT_PATH}
echo "Model save to ${CHECKPOINT_PATH}"


USER_DEFINED_ARGS="--model=meta_gru4rec_ood_if_apg --num_loading_workers=1 --arch_config=${ARCH_CONFIG_CONTENT} --learning_rate=${LEARNING_RATE} \
--max_gradient_norm=${GRADIENT_CLIP} --batch_size=${BATCH_SIZE} --snapshot=${SNAPSHOT} --max_steps=${ITERATION} --checkpoint_dir=${CHECKPOIN\
T_PATH} --arch_config=${ARCH_CONF_FILE}"

#dataset="../../../../data/Amazon_cds/generate_dataset"
#dataset="../../../../data/Amazon_cds/ood_generate_dataset"
#dataset="../../../../data/Amazon_cds/ood_generate_dataset_tiny"
dataset="../../../../data/Amazon_cds/ood_generate_dataset_tiny_10_30u30i"
#dataset="../../../../data/Amazon_cds/ood_generate_dataset_large"
train_file="${dataset}/train.txt"
test_file="${dataset}/test.txt"
data="${train_file},${test_file}"

echo ${USER_DEFINED_ARGS}
python ../../../main_new/multi_metric_meta_ood_baseline_train4.py \
--dataset=${data} \
${USER_DEFINED_ARGS}

echo "Training done: ${CHECKPOINT_PATH}"

