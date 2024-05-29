# Tackling Device Distribution Real-time Shift via Prototype-based Parameter Editing


## Folder Structure

```bash
.
├── Cohn-Kanade-Database #data
│   ├── CK+
│   ├── small
│   ├── large
│   ├── dataset_partition
│   └── ...preprocessed data files
├── Persona_Vision_0 #code
│   ├── checkpoint
│   ├── dataset
│   ├── diagrams
│   ├── main
│   ├── models
│   ├── modules
│   ├── paper_figs
│   ├── scripts
│   ├── statistics
│   └── utils
└── README.md
```

## Data Preprocessing

1. Data preprocessing

```shell
cd Cohn-Kanade-Database
python generate_dataset.py
python generate_dataset_pair.py
```

2. Data Partition

```shell
cd dataset_partition
bash tsne_kmeans_image.sh
python dataset_partition.py
```

## Train


1. Base train

```shell
cd Persona_Vision_0/scripts/facial_expression_recognition/0base/mobilenetv3
bash ck+_base_mobilenetv3_train.sh
```

2. Meta train

```bash
cd Persona_Vision_0/scripts/facial_expression_recognition/2meta/mobilenetv3
bash ck+_meta_mobilenetv3_classifier_train.sh
```

3. Ours train

```bash
cd Persona_Vision_0/scripts/facial_expression_recognition/2meta/mobilenetv3
bash ck+_meta_mobilenetv3_classifier_train_grad.sh
```

## Inference

1. Base inference

```bash
cd Persona_Vision_0/scripts/facial_expression_recognition/0base/mobilenetv3
bash base_test.sh
```

2. Meta inference

```bash
cd Persona_Vision_0/scripts/facial_expression_recognition/2meta/mobilenetv3
bash ck+_meta_mobilenetv3_classifier_test.sh
```

3. Ours inference

```bash
cd Persona_Vision_0/scripts/facial_expression_recognition/2meta/mobilenetv3
bash ck+_meta_mobilenetv3_classifier_test_grad.sh
```

## Train Parameter Editor

> For group training，parameter in shell need to be modified, for instance,
>
> ```bash
> classnum=2  # 2 groups
> cluster=cluster_0  # the 1st group
> size=small  # model=mobilenetv3_small
> ```


1. Finetune on group data

```bash
cd Persona_Vision_0/scripts/facial_expression_recognition/0base/mobilenetv3
bash ck+_base_mobilenetv3_finetune.sh
```

2. Train each group's parameter editor

```bash
cd Persona_Vision_0/scripts/facial_expression_recognition/2meta/mobilenetv3
bash ck+_meta_mobilenetv3_classifier_train_grad_group.sh
```

3. Inference

```bash
cd Persona_Vision_0/scripts/facial_expression_recognition/2meta/mobilenetv3
bash ck+_meta_mobilenetv3_classifier_test_grad_group.sh
```