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

#dataset=$1
#model=$2
##group_num=$3
#cuda_num=$3

class_num=$1
dataset=$2
#group_num=$3
model=$3

python tsne_kmeans.py --class_num ${class_num} --dataset ${dataset} --model ${model}

echo "Training done: ${CHECKPOINT_PATH}"

