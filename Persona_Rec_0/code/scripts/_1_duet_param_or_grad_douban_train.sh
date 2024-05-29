#!/bin/bash
date
#sleep 9600
#sleep 600
#dataset_list=("amazon_beauty" "amazon_cds" "amazon_electronic")
dataset_list=("douban_book" "douban_music")
#dataset_list=("douban_book")
#dataset_list=$("amazon_beauty amazon_cds amazon_electronic")
#dataset_list=("amazon_beauty amazon_cds amazon_electronic")
echo ${dataset_list}
#line_num_list=(7828 21189 30819)
cuda_num_list=(2 3)
#cuda_num_list=(1)
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
#          for type in _3_func_duet_grad_douban_train
#          for type in _3_func_duet_param_douban_train
#          for type in _1_func_duet_grad_douban_train _1_func_duet_grad_no_clip_douban_train _1_func_duet_grad2_douban_train _1_func_duet_grad2_no_clip_douban_train
#          for type in _1_func_duet_grad_no_clip_douban_train _1_func_duet_grad2_douban_train _1_func_duet_grad2_no_clip_douban_train
#          for type in _1_func_duet_grad_douban_train _1_func_duet_grad_no_clip_douban_train _1_func_duet_grad_gru_douban_train _1_func_duet_grad_no_clip_gru_douban_train
#          for type in _1_func_duet_grad_douban_train _1_func_duet_grad_gru_c_douban_train
          for type in _1_func_duet_grad_gru_c_douban_train
#          for type in _1_func_duet_grad_douban_train _1_func_duet_grad_no_clip_douban_train
#          for type in _1_func_duet_grad_gru_douban_train _1_func_duet_grad_no_clip_gru_douban_train
          do
            {
#              bash _2_func_prototype_train.sh ${dataset} ${model} ${group_num} ${cuda_num}
#              bash ${type}.sh ${dataset} ${model} ${group_num} ${cuda_num}
#              for grad_norm_ in 0.5
              for grad_norm in 1.0
              do
                {
                  bash ${type}.sh ${dataset} ${model} ${cuda_num} ${grad_norm}
                } &
              done

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
# bash _1_func_duet_grad_douban_train.sh douban_book din 1
# bash _1_func_duet_grad_no_clip_gru_douban_train.sh douban_book din 0
# bash _1_func_duet_grad2_douban_train.sh douban_music din 2
# bash _1_func_duet_grad2_no_clip_douban_train.sh douban_music din 2
# bash _2_func_prototype_train.sh amazon_beauty din 5 3
# bash _2_func_prototype_train.sh amazon_cds din 5 1
# bash _2_func_prototype_train.sh amazon_electronic sasrec 5 3
# bash _2_func_prototype_train.sh douban_book sasrec 5 3

# bash _2_func_prototype_multi_train.sh amazon_beauty din 5 0

# bash _3_func_duet_param_train.sh amazon_beauty din 0
# bash _3_func_duet_grad_train.sh amazon_beauty din 0