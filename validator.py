import logging

import numpy as np
import torch
from collections import OrderedDict
from mmcv.utils import get_logger
from torch.nn.modules.batchnorm import _BatchNorm
import torch.distributed as dist
import time
from collections.abc import Sequence
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def move_data_to_gpu(cpu_data, gpu_id):
    relocated_data = cpu_data
    if isinstance(cpu_data, Sequence):
        for ind, item in enumerate(cpu_data):
            relocated_data[ind] = move_data_to_gpu(item, gpu_id)
    elif isinstance(cpu_data, dict):
        for key, item in cpu_data.items():
            relocated_data[key] = move_data_to_gpu(item, gpu_id)
    elif isinstance(cpu_data, torch.Tensor):
        if cpu_data.device == torch.device('cpu'):
            return cpu_data.to(gpu_id)
    return relocated_data


class BaseValidator(object):
    def __init__(self, local_rank, logdir, test_loader, model_dict, is_distributed, broadcast_bn_buffer=True, use_ema=False):
        self.local_rank = local_rank
        self.logdir = logdir
        self.is_distributed = is_distributed
        self.use_ema = use_ema
        self.test_loaders = OrderedDict()
        test_loaders = (test_loader,)

        for ind, loader in enumerate(test_loaders):
            tmp_name = loader.dataset.name + '_' + loader.dataset.split
            self.test_loaders[tmp_name] = loader
        self.best_metrics = None
        self.batch_output = {}  # 每一次迭代产生的结果
        self.start_index = 0
        # 设置网络
        self.model_dict = model_dict
        #
        self.val_iter = None
        self.broadcast_bn_buffer = broadcast_bn_buffer

        self.best_acc = 0
        self.best_iteration = 0

    def eval_iter(self, val_batch_data):
        raise NotImplementedError

    def __call__(self, iteration):
        logger = get_logger('basicda', self.logdir, logging.INFO)
        logger.info('start validator')
        self.iteration = iteration
        self.save_flag = False
        for key in self.model_dict:
            self.model_dict[key].eval()
        # 同步bn层的buffer, following mmcv evaluation hook
        if self.is_distributed:
            if self.broadcast_bn_buffer:
                for name, model in self.model_dict.items():
                    for name, module in model.named_modules():
                        if isinstance(module, _BatchNorm) and module.track_running_stats:
                            dist.broadcast(module.running_var, 0)
                            dist.broadcast(module.running_mean, 0)
        # 测试
        for ind, (key, loader) in enumerate(self.test_loaders.items()):
            self.val_dataset_key = key
            time.sleep(2)

            all_feat = []
            all_pred = []
            all_gt = []
            all_out = []
            for val_iter, val_data in enumerate(loader):
                relocated_data = move_data_to_gpu(val_data, self.local_rank)
                self.val_iter = val_iter
                self.batch_output = self.eval_iter(relocated_data)
                self.batch_output.update({'dataset_name': key})
                self.batch_output.update({'dataset_index': ind})

                if self.is_distributed:
                    pred = concat_all_gather(torch.argmax(self.batch_output["pred"], dim=-1))
                    gt = concat_all_gather(self.batch_output["gt"])
                    all_out.append(concat_all_gather(self.batch_output["pred"]))
                    all_feat.append(concat_all_gather(self.batch_output["feat"]))
                else:
                    pred = torch.argmax(self.batch_output["pred"], dim=-1)
                    gt = self.batch_output["gt"]
                    all_out.append(self.batch_output["pred"])
                    all_feat.append(self.batch_output["feat"])
                all_pred.append(pred)
                all_gt.append(gt)
            all_pred = torch.cat(all_pred, dim=0)
            all_gt = torch.cat(all_gt, dim=0)
            all_out = torch.cat(all_out, dim=0)
            all_feat = torch.cat(all_feat, dim=0)
        confusion_mat = confusion_matrix(all_gt.cpu().detach().numpy(), all_pred.cpu().detach().numpy())

        # class-wise acc
        acc = (confusion_mat.diagonal() / confusion_mat.sum(axis=1) * 100)
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        if len(confusion_mat) == 12:
            logger.info(f"avg class-wise acc = {aacc / 100}")
            logger.info(f"class-wise acc {acc}")

        # if self.local_rank == 0:
        #     print(confusion_mat)
        acc = accuracy_score(all_gt.cpu().detach().numpy(), all_pred.cpu().detach().numpy())
        precision = precision_score(all_gt.cpu().detach().numpy(), all_pred.cpu().detach().numpy(), average="macro")
        recall = recall_score(all_gt.cpu().detach().numpy(), all_pred.cpu().detach().numpy(), average="macro")
        F1 = f1_score(all_gt.cpu().detach().numpy(), all_pred.cpu().detach().numpy(), average="macro")
        logger.info("Iteration {}---Acc: {}, Precision: {}, "
                    "Recall: {}, F1: {}".format(self.iteration, round(acc, 4), round(precision, 4),
                                                round(recall, 4), round(F1, 4)))

        if acc > self.best_acc:
            self.best_acc = acc
            self.best_iteration = iteration
        logger.info("Iteration {}, best test acc = {}, "
                    "occurred in {} iterations\n".format(self.iteration, self.best_acc, self.best_iteration))

        return all_feat.detach().cpu(), all_out.detach().cpu(), all_gt.detach().cpu()


class ValidatorSFDAClassRelation(BaseValidator):
    def __init__(self, basic_parameters):
        super(ValidatorSFDAClassRelation, self).__init__(**basic_parameters)

    def eval_iter(self, val_batch_data):
        # val_img, val_label, val_name = val_batch_data
        val_img = val_batch_data['img']
        val_label = val_batch_data['gt_label'].squeeze(1)
        val_metas = val_batch_data['img_metas']
        with torch.no_grad():
            if self.use_ema:
                target_feat, target_logits, feat, pred_unlabeled = self.model_dict['base_model'](val_img)
            else:
                feat, pred_unlabeled, target_feat, target_logits = self.model_dict['base_model'](val_img)
        return {'img': val_img,
                'gt': val_label,
                'img_metas': val_metas,
                'feat': feat,
                'pred': pred_unlabeled,
                'target_pred': target_logits,
                }