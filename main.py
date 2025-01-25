import argparse
import torch
import random
import os
import time
import numpy as np
import torch.distributed as dist
import logging
from copy import deepcopy
from mmcv.utils import get_logger
import torch.nn as nn
from mmcv.runner import get_dist_info
from mmcv.parallel import collate
from mmcv import Config
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from mmcv.parallel import MMDistributedDataParallel

from validator import ValidatorSFDAClassRelation
from trainer import TrainerSFDAClassRelation, deal_with_val_interval, load_pretrained_model
from model import SFDASimplifiedContrastiveModel, BasicModel
from loader import SSDA_CLS_Datasets, SSDA_TEST_Datasets, get_worker_init_fn

Predefined_Control_Keys = ['max_iters', 'log_interval', 'val_interval', 'save_interval', 'max_save_num',
                           'seed', 'cudnn_deterministic', 'pretrained_model', 'checkpoint', 'test_mode',
                           'save_best_model']

parser = argparse.ArgumentParser(description="config")
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--config", type=str, default="./configs/sfda_class_relation/class_relation_officehome_AaD_AC.py")
parser.add_argument("--stop_iteration", type=int, default=-1, help="Stop re-weight according to confidence")
parser.add_argument("--num_k", type=int, default=4)
parser.add_argument("--weight", type=float, default=4.0)
parser.add_argument("--no_fusion", action="store_true")
parser.add_argument("--no_centroid", action="store_true")
parser.add_argument("--kl_weight", type=float, default=0.0)
parser.add_argument("--temp", type=float, default=0.07)
parser.add_argument("--pretrained_model", type=str, default="OfficeHome-Res50/source_only_A.pth")
args = parser.parse_args()

is_distributed = args.local_rank != -1

if is_distributed:
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl")
    local_rank, world_size = get_dist_info()
else:
    local_rank, world_size = 0, 1
    torch.cuda.set_device(0)

cfg = Config.fromfile(args.config)
predefined_keys = ['datasets', 'models', 'control', 'train', 'test']
old_keys = list(cfg._cfg_dict.keys())
for key in old_keys:
    if not key in predefined_keys:
        del cfg._cfg_dict[key]

# check control keys are allowable
control_cfg = cfg.control
for key in control_cfg.keys():
    assert key in Predefined_Control_Keys, '{} is not allowed appeared in control keys'.format(key)

# set default values for control keys
max_iters = control_cfg.get('max_iters', 100000)
log_interval = control_cfg.get('log_interval', 100)
val_interval = control_cfg.get('val_interval', 5000)
save_interval = control_cfg.get('save_interval', 5000)
max_save_num = control_cfg.get('max_save_num', 1)
cudnn_deter_flag = control_cfg.get('cudnn_deterministic', False)
test_mode = control_cfg.get('test_mode', False)
save_best_model = control_cfg.get('save_best_model', True)

# create log dir
run_id = random.randint(1, 100000)
if is_distributed:
    run_id_tensor = torch.ones((1,), device='cuda:{}'.format(local_rank)) * run_id
    torch.distributed.broadcast(run_id_tensor, src=0)
    run_id = int(run_id_tensor.cpu().item())
logdir = os.path.join('runs', os.path.basename(args.config)[:-3], 'exp_' + str(run_id))
if local_rank == 0:
    if not os.path.exists(logdir):
        os.makedirs(logdir)

# create logger
timestamp = time.strftime('runs_%Y_%m%d_%H%M%S', time.localtime())

beta = cfg["train"].get('beta', 0.0)
path_name = f'{os.path.basename(args.config)[:-3]}_num_k_{args.num_k}_weight_{args.weight}_klweight_{args.kl_weight}_temp+{args.temp}_beta_{beta}_stop_{args.stop_iteration}_rank_{local_rank}_{timestamp}.log'
if args.no_centroid:
    path_name = "no_centroid_" + path_name
if args.no_fusion:
    path_name = "no_fusion_" + path_name
log_file = os.path.join(logdir, path_name)

logger = get_logger('basicda', log_file, logging.INFO)
logger.info('log dir is {}'.format(logdir))
logger.info('Let the games begin')

# Setup random seeds, and cudnn_deterministic mode
random_seed = control_cfg.get('seed', None)
logger.info(f'Set random random_seed to {random_seed}, '
            f'deterministic: {cudnn_deter_flag}')
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# build dataloader
dataset_args = deepcopy(cfg["datasets"])
trainset_args = dataset_args['train']
testset_args = dataset_args['test']
n_workers = dataset_args['n_workers']

# train loader
temp_data_builder_args = trainset_args[0].pop('builder', None)
dataset_params = deepcopy(trainset_args[0])
train_dataset = SSDA_CLS_Datasets(dataset_params)
if is_distributed:
    sampler = DistributedSampler(train_dataset, shuffle=True)
    collate_fn = partial(collate, samples_per_gpu=temp_data_builder_args["samples_per_gpu"])
    rank, world_size = get_dist_info()
    worker_init_fn = partial(get_worker_init_fn, num_workers=n_workers, rank=rank, seed=random_seed)
else:
    sampler = RandomSampler(train_dataset)
    collate_fn = partial(collate, samples_per_gpu=temp_data_builder_args["samples_per_gpu"])
    worker_init_fn = None

train_loader = DataLoader(train_dataset, batch_size=temp_data_builder_args["samples_per_gpu"],
                          num_workers=n_workers, shuffle=False,
                          sampler=sampler,
                          drop_last=True, collate_fn=collate_fn,
                          worker_init_fn=worker_init_fn, persistent_workers=True)
logger.info('Train loader has {} images'.format(len(train_loader.dataset)))

# test loader
test_builder_args = testset_args[0].pop('builder', None)
dataset_params = deepcopy(testset_args[0])
test_dataset = SSDA_TEST_Datasets(dataset_params)
if is_distributed:
    sampler = DistributedSampler(test_dataset, shuffle=False)
    collate_fn = partial(collate, samples_per_gpu=test_builder_args["samples_per_gpu"])
    rank, world_size = get_dist_info()
    worker_init_fn = partial(get_worker_init_fn, num_workers=n_workers, rank=rank, seed=random_seed)
else:
    sampler = SequentialSampler(test_dataset)
    collate_fn = partial(collate, samples_per_gpu=test_builder_args["samples_per_gpu"])
    worker_init_fn = None

test_loader = DataLoader(test_dataset, batch_size=test_builder_args["samples_per_gpu"],
                         num_workers=n_workers, shuffle=False,
                         sampler=sampler,
                         drop_last=False, collate_fn=collate_fn,
                         worker_init_fn=worker_init_fn)
logger.info('Test loader has {} images'.format(len(test_loader.dataset)))

# build model and corresponding optimizer, scheduler
model_args = deepcopy(cfg["models"])
shared_lr_scheduler_param = model_args.pop('lr_scheduler', None)
find_unused_parameters = model_args.pop('find_unused_parameters', False)
broadcast_buffers = model_args.pop('broadcast_buffers', False)
sync_bn = model_args.pop('sync_bn', None)

model_args = model_args['base_model']
final_optimizer_args = model_args.get('optimizer', None)
final_lr_scheduler_args = shared_lr_scheduler_param

model = SFDASimplifiedContrastiveModel(model_args["model_dict"], model_args["classifier_dict"], model_args["num_class"],
                                       model_args["low_dim"], model_args["model_moving_average_decay"])
rank, world_size = get_dist_info()
model = model.to('cuda:{}'.format(rank))
if is_distributed:
    model = MMDistributedDataParallel(model, device_ids=[rank],
                                      output_device=rank,
                                      find_unused_parameters=find_unused_parameters,
                                      broadcast_buffers=broadcast_buffers)
    optimizer_args = deepcopy(final_optimizer_args)
    logger.info('Use optim_parameters within the model')
    optim_model_param = model.module.optim_parameters(lr=optimizer_args["lr"])
else:
    optimizer_args = deepcopy(final_optimizer_args)
    logger.info('Use optim_parameters within the model')
    optim_model_param = model.optim_parameters(lr=optimizer_args["lr"])

optimizer_args['params'] = optim_model_param
optimizer = torch.optim.SGD(params=optimizer_args["params"], lr=optimizer_args["lr"],
                            momentum=optimizer_args["momentum"],
                            weight_decay=optimizer_args["weight_decay"], nesterov=optimizer_args["nesterov"])
model_dict = nn.ModuleDict()
optimizer_dict = {}
model_dict["base_model"] = model
optimizer_dict["base_model"] = optimizer

# build source model and optimizer
source_model = BasicModel(model_args["model_dict"], model_args["classifier_dict"])
rank, world_size = get_dist_info()
source_model = source_model.to('cuda:{}'.format(rank))
if is_distributed:
    source_model = MMDistributedDataParallel(source_model, device_ids=[rank],
                                             output_device=rank,
                                             find_unused_parameters=find_unused_parameters,
                                             broadcast_buffers=broadcast_buffers)
    optimizer_args = deepcopy(final_optimizer_args)
    logger.info('Use optim_parameters within the model')
    optim_model_param = source_model.module.optim_parameters(lr=optimizer_args["lr"])
else:
    optimizer_args = deepcopy(final_optimizer_args)
    logger.info('Use optim_parameters within the model')
    optim_model_param = source_model.optim_parameters(lr=optimizer_args["lr"])

optimizer_args['params'] = optim_model_param
source_optimizer = torch.optim.SGD(params=optimizer_args["params"], lr=optimizer_args["lr"],
                                   momentum=optimizer_args["momentum"],
                                   weight_decay=optimizer_args["weight_decay"], nesterov=optimizer_args["nesterov"])
source_model_dict = nn.ModuleDict()
source_optimizer_dict = {}
source_model_dict["base_model"] = source_model
source_optimizer_dict["base_model"] = source_optimizer

# build trainer
beta = cfg["train"].get('beta', 0.0)
trainer = TrainerSFDAClassRelation(rank, model_dict, optimizer_dict, source_model_dict, source_optimizer_dict,
                                   train_loader, logdir, is_distributed, cfg["train"]['pseudo_update_interval'],
                                   beta, args.num_k, args.weight, args.kl_weight, args.temp, args.stop_iteration,
                                   args.no_fusion, args.no_centroid, max_iters=max_iters)
trained_iteration = 0

# load pretrained weights
pretrained_model = args.pretrained_model
logger.info('Load pretrained model in {}'.format(pretrained_model))
trainer.load_pretrained_model(pretrained_model, is_distributed)

load_pretrained_model(source_model_dict, pretrained_model, is_distributed)

# build validator
basic_parameters = {
    "local_rank": rank,
    "logdir": logdir,
    "test_loader": test_loader,
    "model_dict": model_dict,
    "broadcast_bn_buffer": True,
    "is_distributed": is_distributed
}
validator = ValidatorSFDAClassRelation(basic_parameters)

basic_parameters = {
    "local_rank": rank,
    "logdir": logdir,
    "test_loader": test_loader,
    "model_dict": model_dict,
    "broadcast_bn_buffer": True,
    "is_distributed": is_distributed,
    "use_ema": True
}
ema_validator = ValidatorSFDAClassRelation(basic_parameters)

source_basic_parameters = {
    "local_rank": rank,
    "logdir": logdir,
    "test_loader": test_loader,
    "model_dict": source_model_dict,
    "broadcast_bn_buffer": True,
    "is_distributed": is_distributed
}
source_validator = ValidatorSFDAClassRelation(source_basic_parameters)

# deal with val_interval
val_point_list = deal_with_val_interval(val_interval, max_iters=max_iters, trained_iteration=trained_iteration)

# start training and testing
last_val_point = trained_iteration


class EarlyStopping(object):
    def __init__(self):
        self.patience = 5
        self.best_acc = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, acc):
        if self.best_acc is None:
            self.best_acc = acc
        elif acc > self.best_acc:
            self.best_acc = acc
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                logger.info('Early stopping')
                self.early_stop = True


early_stopping = EarlyStopping()
for val_point in val_point_list:
    # train
    trainer(train_iteration=val_point - last_val_point)
    time.sleep(2)

    logger.info("Target Model!!!")
    feat, out, all_gt = validator(trainer.iteration)
    logger.info("EMA Model!!!")
    _, _, _ = ema_validator(trainer.iteration)
    logger.info("Source Model!!!")
    feat_source, out_source, _ = source_validator(trainer.iteration)

    accuracy = torch.mean((torch.argmax(nn.Softmax(-1)(out), dim=-1) == all_gt).float()).item()
    early_stopping(accuracy)
    if early_stopping.early_stop:
        break

    last_val_point = val_point
