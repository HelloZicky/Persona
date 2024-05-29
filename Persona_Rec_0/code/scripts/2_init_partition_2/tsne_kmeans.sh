date
#wait 3600
#for eps in 0.005 0.01 0.02 0.03 0.05 0.1 0.2
#for eps in 0.005 0.01 0.02 0.03 0.05
#for eps in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.12 0.15 0.18 0.2 0.22 0.25 0.28 0.3 0.4
#for eps in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2
#for class_num in 10 15 20 25 30 35 40 45 50 55 60 70 80 85 90 100 171 180 200 250 300 350 400 450 500
#for class_num in 85 171
#for class_num in 2 3 5 10 20 30 50 100 200 300 500
for dataset in amazon_beauty amazon_cds amazon_electronic movielens_1m movielens_100k douban_book douban_music
#for dataset in douban_music
#for dataset in amazon_beauty amazon_cds amazon_electronic movielens_1m movielens_100k douban_music
do
  {
    for model in din gru4rec sasrec
    do
      {
#        for class_num in 2 3 5 10 20 30 50 100
#        for class_num in 30 50 100
        for class_num in 2 3 5 10 20
#        for class_num in 2
        do
          {
            echo "class_num=${class_num}"
#            python tsne_kmeans.py --class_num ${class_num} --dataset ${dataset} --model ${model}
            bash func_tsne_kmeans.sh ${class_num} ${dataset} ${model}
          } &
        done
      } &
    done
  } &
done
#wait # 等待所有任务结束
date

# python tsne_kmeans.py --class_num 5 --dataset movielens_100k --model din
# python tsne_kmeans.py --class_num ${class_num} --dataset ${dataset} --model ${model}