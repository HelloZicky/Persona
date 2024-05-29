date
#sleep 15000
for((i=1;i<=5;i++))
do
{
  for model in new_din new_gru4rec new_sasrec
    do
    {
      for type in multi_metric_meta_ood_uncertainty
      do
        {
          folder="${model}/${type}"
          echo ${folder}
          cd ${folder}
          if [ ${type} == "multi_metric_meta_30" ] || [ ${type} == "multi_metric_meta_ood_gru" ];then
#            for file in amazon_electronic_train amazon_electronic_ood_train amazon_electronic_ood2_train
            for file in amazon_electronic_train amazon_electronic_ood_train  # 不要用ood2, ood2使用了target item
            do
              {
                sh "${file}.sh"
              }&
             done

          elif [ ${type} == "multi_metric_meta_random_request" ];then
            sh amazon_electronic_train.sh

          elif [ ${type} == "multi_metric_meta_ood_baseline" ];then
            for file in amazon_electronic_ood_lof_train amazon_electronic_ood_ocsvm_train # 不要用ood2, ood2使用了target item
            do
              {
                sh "${file}.sh"
              }&
             done

          else
            sh amazon_electronic_ood_train.sh
          fi
          cd ../../
        } &
      done
    } &
    done
    wait
#    sleep 960
#    sleep 2100
    sleep 4200
}
done
wait # 等待所有任务结束
date
