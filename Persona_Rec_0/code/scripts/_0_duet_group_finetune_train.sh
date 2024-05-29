#!/bin/bash
date
#sleep 3600
dataset_list=("amazon_beauty" "amazon_cds" "amazon_electronic")
#dataset_list=("amazon_cds")
#dataset_list=("amazon_cds" "amazon_electronic")
#dataset_list=("movielens_100k" "movielens_1m" "movielens_10m")
#dataset_list=("movielens_100k" "movielens_1m" "douban_book" "douban_music")
#dataset_list=("movielens_1m" "douban_book" "douban_music" "movielens_100k")
#dataset_list=("douban_book" "douban_music" "movielens_1m")
#dataset_list=("movielens_1m" "douban_book" "douban_music")
#dataset_list=("movielens_100k")
#dataset_list=("amazon_beauty" "amazon_cds" "amazon_electronic")
#dataset_list=("douban_book" "douban_movie")
#dataset_list=$("amazon_beauty amazon_cds amazon_electronic")
#dataset_list=("amazon_beauty amazon_cds amazon_electronic")
echo ${dataset_list}
line_num_list=(7828 21189 30819)
#cuda_num_list=(1 2 3)
cuda_num_list=(2)
#cuda_num_list=(7)
#cuda_num_list=(0 4 5)
#cuda_num_list=(2 3 5)
#cuda_num_list=(1)
#cuda_num_list=(0 1)
#line_num_list=$("7828 21189 30819")
echo ${line_num_list}
length=${#dataset_list[@]}
#for dataset line_num in zip()
for ((i=0; i<${length}; i++));
#for i in {0..${length}-1};
do
{
    dataset=${dataset_list[i]}
#    line_num=${line_num_list[i]}
    cuda_num=${cuda_num_list[i]}
    for model in din gru4rec sasrec
#    for model in gru4rec sasrec
#    for model in sasrec
    do
    {
#      for class_num in 5 10 50 100 500
#      for class_num in 5 10 50
#      for class_num in 100 500
#      for group_num in 5 10 50 100
#      for group_num in 5 10 50
#      for group_num in 2 3 5 10
#      for group_num in 3 5 10
#      for group_num in 2 3 5 10
#      for group_num in 2 3
      for group_num in 5 10
#      for group_num in 2
#      for group_num in 3
      do
        {
#          for type in _2_func_prototype_train _2_func_prototype_multi_train
#          for type in _2_func_prototype_multi_train
#          for type in _0_func_duet_train _0_func_base_train
#          for type in _0_func_base_movielens_train _0_func_duet_movielens_train
#          for type in _0_func_finetune_movielens_train
#          for type in _0_func_group_finetune_movielens_train
          for train_mode in long_tail meta_grad meta_grad_gru
          do
            {
              for type in _0_func_group_finetune_train
              do
                {
    #              bash _2_func_prototype_train.sh ${dataset} ${model} ${group_num} ${cuda_num}
#                  bash ${type}.sh ${dataset} ${model} ${group_num} ${cuda_num}
                  bash ${type}.sh ${dataset} ${model} ${group_num} ${cuda_num} ${train_mode}
    #              bash ${type}.sh ${dataset} ${model} ${group_num} 7
    #              bash ${type}.sh ${dataset} ${model} ${group_num} 0
    #              bash ${type}.sh ${dataset} ${model} ${cuda_num}
    #              bash ${type}.sh ${dataset} ${model} 0
                } &
              done
            } &
            done
        } &
        done
    } &
    done
} &
done
wait # 等待所有任务结束
date
# bash _0_func_group_finetune_train.sh amazon_electronic sasrec 10 3 long_tail
# bash _0_func_group_finetune_train.sh amazon_beauty din 5 3 meta_grad
# bash _0_func_group_finetune_train.sh amazon_beauty din 5 3 meta_grad_gru
# bash _2_func_prototype_train.sh amazon_beauty din 5 3
# bash _2_func_prototype_train.sh amazon_cds din 5 1
# bash _2_func_prototype_train.sh amazon_electronic sasrec 5 3
# bash _2_func_prototype_train.sh douban_book sasrec 5 3

# bash _2_func_prototype_multi_train.sh amazon_beauty din 5 0
# bash _0_func_duet_train.sh amazon_beauty din 0