#!/bin/bash
date
dataset_list=("amazon_beauty" "amazon_cds" "amazon_electronic")
#dataset_list=$("amazon_beauty amazon_cds amazon_electronic")
#dataset_list=("amazon_beauty amazon_cds amazon_electronic")
echo ${dataset_list}
line_num_list=(7828 21189 30819)
cuda_num_list=(1 2 3)
#line_num_list=$("7828 21189 30819")
echo ${line_num_list}
length=${#dataset_list[@]}
#for dataset line_num in zip()
for ((i=0; i<${length}; i++));
#for i in {0..${length}-1};
do
{
    dataset=${dataset_list[i]}
    line_num=${line_num_list[i]}
#    cuda_num=${i+1}
    cuda_num=${cuda_num_list[i]}
    for model in din gru4rec sasrec
#    for model in gru4rec sasrec
    do
    {
#      for class_num in 5 10 50 100 500
#      for class_num in 5 10 50
      for class_num in 100 500
      do
        {
          # for type in multi_metric_base_30
          for type in _deep_cluster
          do
            {
#              folder="new_${model}/${type}"
#              echo ${folder}
#              cd ${folder}
              # sh amazon_beauty_train.sh
#              sh amazon_beauty_ood_train.sh
              sh _func_deep_cluster.sh ${dataset} ${model} ${class_num} ${line_num} ${cuda_num}
#              cd ../../
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
# bash _func_deep_cluster.sh amazon_beauty din 10 7828 3
# bash _func_deep_cluster.sh amazon_cds din 10 21189 3
# bash _func_deep_cluster.sh amazon_electronic sasrec 10 30819 3