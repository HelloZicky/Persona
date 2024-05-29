#!/bin/bash
date
dataset_list=("amazon_beauty" "amazon_cds" "amazon_electronic")
#dataset_list=("amazon_cds" "amazon_electronic")
#dataset_list=("amazon_beauty")
#dataset_list=("douban_book" "douban_movie")
#dataset_list=$("amazon_beauty amazon_cds amazon_electronic")
#dataset_list=("amazon_beauty amazon_cds amazon_electronic")
echo ${dataset_list}
line_num_list=(7828 21189 30819)
#cuda_num_list=(0 1 2 3)
cuda_num_list=(1 2 3)
#cuda_num_list=(3)
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
#    for model in din
    do
    {
#      for class_num in 5 10 50 100 500
#      for class_num in 5 10 50
#      for class_num in 100 500
#      for group_num in 5 10 50 100
#      for group_num in 5 10 50
#      for group_num in 5 10
#      do
#        {
#          for type in _2_func_prototype_train _2_func_prototype_multi_train
#          for type in _2_func_prototype_multi_train
#          for type in _3_func_duet_param_train _3_func_duet_grad_train
#          for type in _3_func_duet_grad_train
#          for type in _3_func_duet_param_train
#          for type in _2_func_duet_grad_pred
#          for type in _2_func_duet_grad_gru_pred
          for type in _2_func_duet_grad_pred _2_func_duet_grad_gru_pred
          do
            {
#              bash _2_func_prototype_train.sh ${dataset} ${model} ${group_num} ${cuda_num}
#              bash ${type}.sh ${dataset} ${model} ${group_num} ${cuda_num}
              bash ${type}.sh ${dataset} ${model} ${cuda_num}
            } &
          done
#        } &
#        done
    } &
    done
} &
done
wait # 等待所有任务结束
date
# bash _2_func_duet_grad_gru_pred.sh amazon_beauty din 1
# bash _2_func_duet_grad_pred.sh amazon_beauty din 1
# bash _2_func_duet_grad_pred.sh amazon_beauty din 0

# bash _2_func_prototype_train.sh amazon_beauty din 5 3
# bash _2_func_prototype_train.sh amazon_cds din 5 1
# bash _2_func_prototype_train.sh amazon_electronic sasrec 5 3
# bash _2_func_prototype_train.sh douban_book sasrec 5 3

# bash _2_func_prototype_multi_train.sh amazon_beauty din 5 0

# bash _3_func_duet_param_train.sh amazon_beauty din 0
# bash _3_func_duet_grad_train.sh amazon_beauty din 0