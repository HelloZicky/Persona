#!/bin/bash
date
#sleep 300
#dataset_list=("amazon_beauty" "amazon_cds" "amazon_electronic")
dataset_list=("amazon_beauty")
#dataset_list=("amazon_cds" "amazon_electronic")
#dataset_list=("amazon_electronic")
#dataset_list=("amazon_electronic")
#dataset_list=("amazon_beauty")
#dataset_list=("amazon_beauty" "amazon_cds")
#dataset_list=("douban_book" "douban_music")
#dataset_list=$("amazon_beauty amazon_cds amazon_electronic")
#dataset_list=("amazon_beauty amazon_cds amazon_electronic")
echo ${dataset_list}
line_num_list=(7828 21189 30819)
#cuda_num_list=(0 1 2 3)
#cuda_num_list=(2 3)
#cuda_num_list=(1 2 3)
#cuda_num_list=(0 1 2)
cuda_num_list=(2)
#cuda_num_list=(1 6 7)
#cuda_num_list=(2 3 4)
#cuda_num_list=(0 4 5)
#cuda_num_list=(0 2)
#cuda_num_list=(0 3 5)
#cuda_num_list=(1)
#cuda_num_list=(0 1 3)
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
#    for model in din gru4rec sasrec
    for model in gru4rec sasrec
#    for model in sasrec
#    for model in din
    do
    {
#      for class_num in 5 10 50 100 500
#      for class_num in 5 10 50
#      for class_num in 100 500
#      for group_num in 5 10 50 100
#      for group_num in 5 10 50
#      for group_num in 2 3 5 10 20 30 50 100
#      for group_num in 2 3 5 10
#      for group_num in 2 3 5
#      for group_num in 5
#      for group_num in 2 3
#      for group_num in 5 10
#      for group_num in 5
      for group_num in 2
#      for group_num in 20 30 50 100
#      for group_num in 5
#      for group_num in 20 30
#      for group_num in 5 10
#      for group_num in 5
      do
        {
#          for type in _2_func_prototype_train _2_func_prototype_multi_train
#          for type in _2_func_prototype_multi_train
#          for type in _3_func_duet_param_train _3_func_duet_grad_train
#          for type in _3_func_duet_grad_train
#          for type in _3_func_duet_param_train
#          for type in _3_1_func_group_duet_grad_train _3_1_func_group_duet_grad_gru_train _3_1_func_group_duet_grad_gru_center_train
#          for type in _3_1_func_group_duet_grad_train _3_1_func_group_duet_grad_gru_train
#          for type in _3_1_func_group_duet_grad_train
#          for type in _3_1_func_group_duet_grad_gru_train
#          for type in _3_1_func_group_duet_grad_gru_center_train
          for type in _3_1_func_group_duet_grad_gru_center_pred
#          for type in _3_1_func_group_duet_grad_gru_center_dynamic_train
#          for type in _3_1_func_group_duet_grad_douban_train
#          for type in _3_1_func_group_duet_param_train
#          for type in _3_1_func_group_duet_param_douban_train
          do
            {
#              bash _2_func_prototype_train.sh ${dataset} ${model} ${group_num} ${cuda_num}
#              bash ${type}.sh ${dataset} ${model} ${group_num} ${cuda_num}
#              for grad_norm in 0.1
#              for grad_norm in 0.5
#              for grad_norm in 1.0
#              for grad_norm in 5.0
##              for grad_norm in 0.1 0.2 0.5 1.0
#              do
#                {
#                  bash ${type}.sh ${dataset} ${model} ${group_num} ${cuda_num} ${grad_norm}
##                  bash ${type}.sh ${dataset} ${model} ${group_num} 7 ${grad_norm}
#                } &
#              done
#                for train_mode in long_tail meta_grad meta_grad_gru
#                for train_mode in long_tail meta_grad
                for train_mode in meta_grad_gru
                do
                  {
                    for grad_norm in 1.0
                    do
                      {
                        bash ${type}.sh ${dataset} ${model} ${group_num} ${cuda_num} ${grad_norm} ${train_mode}
#                        bash ${type}.sh ${dataset} ${model} ${group_num} 0 ${grad_norm} ${train_mode}
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
} &
done
wait # 等待所有任务结束
date
# bash _3_1_func_group_duet_grad_gru_center_train.sh amazon_beauty sasrec 5 2 1.0 long_tail
# bash _3_1_func_group_duet_grad_gru_center_train.sh amazon_beauty sasrec 5 2 1.0 grad_meta
# bash _3_1_func_group_duet_grad_gru_center_train.sh amazon_beauty sasrec 5 2 1.0 grad_meta_gru
# bash _3_1_func_group_duet_grad_gru_center_train.sh amazon_beauty sasrec 5 2 1.0
# bash _3_1_func_group_duet_grad_gru_center_train.sh amazon_beauty sasrec 5 2 0.1
# bash _3_1_func_group_duet_grad_gru_center_train.sh amazon_cds sasrec 10 2 0.1
# bash _3_1_func_group_duet_grad_gru_center_train.sh amazon_cds din 5 2 0.1
# bash _3_1_func_group_duet_grad_gru_center_train.sh amazon_electronic din 5 3 0.1
# bash _3_1_func_group_duet_grad_gru_center_train.sh amazon_electronic sasrec 10 3 0.1
#for ((i=0; i<${length}; i++));
##for i in {0..${length}-1};
#do
#{
#    dataset=${dataset_list[i]}
##    line_num=${line_num_list[i]}
#    cuda_num=${cuda_num_list[i]}
#    for model in din gru4rec sasrec
##    for model in gru4rec sasrec
##    for model in din
#    do
#    {
##      for class_num in 5 10 50 100 500
##      for class_num in 5 10 50
##      for class_num in 100 500
##      for group_num in 5 10 50 100
##      for group_num in 5 10 50
##      for group_num in 2 3 5 10 20 30 50 100
##      for group_num in 2 3 5 10
##      for group_num in 2 3
#      for group_num in 5 10
##      for group_num in 20 30 50 100
##      for group_num in 5
##      for group_num in 20 30
##      for group_num in 5 10
##      for group_num in 5
#      do
#        {
##          for type in _2_func_prototype_train _2_func_prototype_multi_train
##          for type in _2_func_prototype_multi_train
##          for type in _3_func_duet_param_train _3_func_duet_grad_train
##          for type in _3_func_duet_grad_train
##          for type in _3_func_duet_param_train
##          for type in _3_1_func_group_duet_grad_train _3_1_func_group_duet_grad_gru_train _3_1_func_group_duet_grad_gru_center_train
##          for type in _3_1_func_group_duet_grad_train _3_1_func_group_duet_grad_gru_train
##          for type in _3_1_func_group_duet_grad_train
##          for type in _3_1_func_group_duet_grad_gru_train
#          for type in _3_1_func_group_duet_grad_gru_center_train
##          for type in _3_1_func_group_duet_grad_gru_center_dynamic_train
##          for type in _3_1_func_group_duet_grad_douban_train
##          for type in _3_1_func_group_duet_param_train
##          for type in _3_1_func_group_duet_param_douban_train
#          do
#            {
##              bash _2_func_prototype_train.sh ${dataset} ${model} ${group_num} ${cuda_num}
##              bash ${type}.sh ${dataset} ${model} ${group_num} ${cuda_num}
#              for grad_norm_ in 0.5
##              for grad_norm_ in 1.0
##              for grad_norm_ in 0.1 0.2 0.5 1.0
#              do
#                {
#                  bash ${type}.sh ${dataset} ${model} ${group_num} ${cuda_num} ${grad_norm_}
#                } &
#              done
#
#            } &
#          done
#        } &
#        done
#    } &
#    done
#} &
#done
#wait # 等待所有任务结束
#date

# bash _3_1_func_group_duet_grad_gru_center_train.sh amazon_cds din 10 2 1.0
# bash _3_1_func_group_duet_grad_gru_center_train.sh amazon_beauty din 10 0 0.5
# bash _3_1_func_group_duet_grad_gru_center_train.sh amazon_beauty sasrec 10 0 0.5
# bash _3_1_func_group_duet_grad_gru_center_train.sh amazon_beauty din 2 0 0.5
# bash _3_1_func_group_duet_grad_gru_center_dynamic_train.sh amazon_beauty din 2 0
# bash _3_1_func_group_duet_grad_gru_center_dynamic_train.sh amazon_beauty din 20 0
# bash _2_func_prototype_train.sh amazon_cds din 20 0
# bash _2_func_prototype_train.sh amazon_cds din 30 0
# bash _2_func_prototype_train.sh amazon_electronic sasrec 20 0
# bash _2_func_prototype_train.sh amazon_electronic sasrec 30 0

# bash _3_1_func_group_duet_grad_train.sh amazon_cds din 20 0
# bash _3_1_func_group_duet_grad_train.sh amazon_cds din 30 0
# bash _3_1_func_group_duet_grad_train.sh amazon_electronic sasrec 20 0
# bash _3_1_func_group_duet_grad_train.sh amazon_electronic sasrec 30 0

# bash _2_func_prototype_train.sh amazon_beauty din 5 3
# bash _2_func_prototype_train.sh amazon_cds din 5 1
# bash _2_func_prototype_train.sh amazon_electronic sasrec 5 3
# bash _2_func_prototype_train.sh douban_book sasrec 5 3

# bash _2_func_prototype_multi_train.sh amazon_beauty din 5 0

# bash _3_func_duet_param_train.sh amazon_beauty din 0
# bash _3_func_duet_grad_train.sh amazon_beauty din 0
# bash _3_1_func_group_duet_grad_train.sh amazon_beauty din 5 3