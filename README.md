# BN-SFDA

This reposity is official implementation of

## 1. Preparation
``` shell

```

## 2. Dataset And Model
The datasets (i.e., Office-31, Office-home, and VisDA-C) and corresponding source models are availabel at https://zenodo.org/records/14722689.

## 3. Code
### 3.1 Office-Home
``` shell
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --config ./configs/sfda_class_relation/class_relation_officehome_AaD_AC.py --pretrained_model OfficeHome-Res50/source_only_A.pth --num_k 4 --stop_iteration 5000
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --config ./configs/sfda_class_relation/class_relation_officehome_AaD_AP.py --pretrained_model OfficeHome-Res50/source_only_A.pth --num_k 4 --stop_iteration 5000
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --config ./configs/sfda_class_relation/class_relation_officehome_AaD_AR.py --pretrained_model OfficeHome-Res50/source_only_A.pth --num_k 4 --stop_iteration 5000

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --config ./configs/sfda_class_relation/class_relation_officehome_AaD_CA.py --pretrained_model OfficeHome-Res50/source_only_C.pth --num_k 4 --stop_iteration 5000
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --config ./configs/sfda_class_relation/class_relation_officehome_AaD_CP.py --pretrained_model OfficeHome-Res50/source_only_C.pth --num_k 4 --stop_iteration 5000
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --config ./configs/sfda_class_relation/class_relation_officehome_AaD_CR.py --pretrained_model OfficeHome-Res50/source_only_C.pth --num_k 4 --stop_iteration 5000

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --config ./configs/sfda_class_relation/class_relation_officehome_AaD_PA.py --pretrained_model OfficeHome-Res50/source_only_P.pth --num_k 4 --stop_iteration 5000
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --config ./configs/sfda_class_relation/class_relation_officehome_AaD_PC.py --pretrained_model OfficeHome-Res50/source_only_P.pth --num_k 4 --stop_iteration 5000
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --config ./configs/sfda_class_relation/class_relation_officehome_AaD_PR.py --pretrained_model OfficeHome-Res50/source_only_P.pth --num_k 4 --stop_iteration 5000

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --config ./configs/sfda_class_relation/class_relation_officehome_AaD_RA.py --pretrained_model OfficeHome-Res50/source_only_R.pth --num_k 4 --stop_iteration 5000
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --config ./configs/sfda_class_relation/class_relation_officehome_AaD_RC.py --pretrained_model OfficeHome-Res50/source_only_R.pth --num_k 4 --stop_iteration 5000
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --config ./configs/sfda_class_relation/class_relation_officehome_AaD_RP.py --pretrained_model OfficeHome-Res50/source_only_R.pth --num_k 4 --stop_iteration 5000
```

### 3.2 VisDA-C
``` shell

```
