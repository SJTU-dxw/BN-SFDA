set -e

# visda
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --config ./configs/sfda_visda/class_relation_visda_AaD_2gpu.py --pretrained_model VisDA/source_only_visda.pth

# office-home
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --config ./configs/sfda_class_relation/class_relation_officehome_AaD_AC.py --pretrained_model OfficeHome-Res50/source_only_A.pth
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --config ./configs/sfda_class_relation/class_relation_officehome_AaD_AP.py --pretrained_model OfficeHome-Res50/source_only_A.pth
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --config ./configs/sfda_class_relation/class_relation_officehome_AaD_AR.py --pretrained_model OfficeHome-Res50/source_only_A.pth

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --config ./configs/sfda_class_relation/class_relation_officehome_AaD_CA.py --pretrained_model OfficeHome-Res50/source_only_C.pth
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --config ./configs/sfda_class_relation/class_relation_officehome_AaD_CP.py --pretrained_model OfficeHome-Res50/source_only_C.pth
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --config ./configs/sfda_class_relation/class_relation_officehome_AaD_CR.py --pretrained_model OfficeHome-Res50/source_only_C.pth

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --config ./configs/sfda_class_relation/class_relation_officehome_AaD_PA.py --pretrained_model OfficeHome-Res50/source_only_P.pth
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --config ./configs/sfda_class_relation/class_relation_officehome_AaD_PC.py --pretrained_model OfficeHome-Res50/source_only_P.pth
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --config ./configs/sfda_class_relation/class_relation_officehome_AaD_PR.py --pretrained_model OfficeHome-Res50/source_only_P.pth

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --config ./configs/sfda_class_relation/class_relation_officehome_AaD_RA.py --pretrained_model OfficeHome-Res50/source_only_R.pth
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --config ./configs/sfda_class_relation/class_relation_officehome_AaD_RC.py --pretrained_model OfficeHome-Res50/source_only_R.pth
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py --config ./configs/sfda_class_relation/class_relation_officehome_AaD_RP.py --pretrained_model OfficeHome-Res50/source_only_R.pth
