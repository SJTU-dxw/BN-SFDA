from mmcv.runner import get_dist_info
import torch
import time
from collections.abc import Sequence
import torch.nn.functional as F
from mmcv.utils import get_logger
import logging
import numpy as np
from scipy.spatial.distance import cdist
from collections import OrderedDict
from metric import Metric


def load_pretrained_model(model_dict, weights_path, is_distributed):
    weights = torch.load(weights_path, map_location='cpu')
    weights = weights['base_model']
    for key in weights:
        key_split = key.split('.')
        if key_split[1] in ['target_network', 'target_classifier']:
            key_split[1] = key_split[1].replace('target', 'online')
            online_key = '.'.join(key_split)
            weights[key] = weights[online_key]
    if not is_distributed:
        new_weights = OrderedDict()
        for key in weights:
            new_key = key.replace("module.", "")
            new_weights[new_key] = weights[key]
        model_dict['base_model'].load_state_dict(new_weights, strict=False)
    else:
        new_weights = OrderedDict()
        for key in weights:
            if "module." not in key:
                new_key = "module." + key  # key.replace("module.", "")
            else:
                new_key = key
            new_weights[new_key] = weights[key]
        resp = model_dict['base_model'].load_state_dict(new_weights, strict=False)


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


class TrainerSFDAClassRelation(object):
    def __init__(self, local_rank, model_dict, optimizer_dict, source_model_dict, source_optimizer_dict,
                 train_loader, logdir, is_distributed, pseudo_update_interval, beta, num_k, weight, kl_weight, temp,
                 stop_iteration, no_fusion, no_centroid, feat_dim=256, bank_size=512, max_iters=15000):
        self.local_rank = local_rank
        self.model_dict = model_dict
        self.optimizer_dict = optimizer_dict
        self.train_loaders = (train_loader,)
        self.train_loader_iterator = [item.__iter__() for item in self.train_loaders]
        self.train_loader_epoch_count = [0 for i in range(len(self.train_loader_iterator))]
        self.num_class = self.train_loaders[0].dataset.n_classes
        self.source_model_dict = source_model_dict
        self.source_optimizer_dict = source_optimizer_dict

        self.logdir = logdir
        self.max_iters = max_iters
        self.is_distributed = is_distributed

        base_model = self.model_dict['base_model']
        if self.is_distributed:
            fc_weight = base_model.module.online_classifier.fc.weight_v.detach()
        else:
            fc_weight = base_model.online_classifier.fc.weight_v.detach()
        normalized_fc_weight = F.normalize(fc_weight)
        self.prototypes = normalized_fc_weight.detach().clone()

        base_model = self.source_model_dict['base_model']
        if is_distributed:
            for param in base_model.module.online_classifier.parameters():
                param.requires_grad = False
        else:
            for param in base_model.online_classifier.parameters():
                param.requires_grad = False

        # fix classifier
        base_model = self.model_dict['base_model']
        if is_distributed:
            for param in base_model.module.online_classifier.parameters():
                param.requires_grad = False
        else:
            for param in base_model.online_classifier.parameters():
                param.requires_grad = False

        self.log_interval = 100
        self.lambda_aad = 1.0
        self.prob_threshold = 0.95
        self.lambda_nce = 1.0
        self.lambda_temp = temp
        self.lambda_fixmatch_temp = temp
        self.pseudo_update_interval = pseudo_update_interval
        self.threshold = 0
        self.lambda_fixmatch = 1.0
        self.bank_size = 512
        self.non_diag_alpha = 1.0
        self.beta = beta

        self.iteration = 0
        self.train_batch_output = {}
        self.metric = Metric()

        rank, world_size = get_dist_info()
        self.world_size = world_size

        num_image = len(self.train_loaders[0].dataset)
        self.weak_feat_bank = F.normalize(torch.randn(num_image, feat_dim).to('cuda:{}'.format(rank)))
        self.weak_score_bank = torch.randn(num_image, self.num_class).to('cuda:{}'.format(rank))
        self.label_bank = torch.zeros((num_image,), dtype=torch.long).to('cuda:{}'.format(rank))
        self.class_prototype_bank = F.normalize(torch.randn(self.num_class, feat_dim).to('cuda:{}'.format(rank)))
        self.aad_weak_feat_bank = F.normalize(torch.randn(num_image, feat_dim).to('cuda:{}'.format(rank)))
        self.aad_weak_score_bank = torch.randn(num_image, self.num_class).to('cuda:{}'.format(rank))

        self.source_aad_weak_feat_bank = F.normalize(torch.randn(num_image, feat_dim).to('cuda:{}'.format(rank)))
        self.source_aad_weak_score_bank = torch.randn(num_image, self.num_class).to('cuda:{}'.format(rank))
        #
        self.weak_negative_bank = torch.randn(bank_size, self.num_class).to('cuda:{}'.format(rank))
        self.weak_negative_bank_ptr = torch.zeros(1, dtype=torch.long).to('cuda:{}'.format(rank))
        self.strong_negative_bank = torch.randn(bank_size, self.num_class).to('cuda:{}'.format(rank))
        self.strong_negative_bank_ptr = torch.zeros(1, dtype=torch.long).to('cuda:{}'.format(rank))

        self.scaler = torch.cuda.amp.GradScaler()
        self.scaler_source = torch.cuda.amp.GradScaler()

        self.memory_size = 512
        self.num_k = num_k
        self.weight = weight  # Use for Centroid
        self.kl_weight = kl_weight  # Use for COnsistency Loss
        self.fusion_flag = not no_fusion  # Use Source Model
        self.use_centroid = not no_centroid  # Use Centroid
        self.stop_iteration = stop_iteration
        self.second_stage = False

    def set_train_state(self):
        for name in self.model_dict.keys():
            self.model_dict[name].train()

    def set_eval_state(self):
        for name in self.model_dict.keys():
            self.model_dict[name].eval()

    def set_epoch(self, ind):
        assert hasattr(self.train_loaders[ind].sampler, 'set_epoch'), 'sampler of dataloader {} has not set_epoch func'
        logger = get_logger('basicda', self.logdir, logging.INFO)
        self.train_loader_epoch_count[ind] += 1
        tmp_epoch = self.train_loader_epoch_count[ind]
        self.train_loaders[ind].sampler.set_epoch(tmp_epoch)
        logger.info("set_epoch of Dataloader {}, param {}".format(ind, tmp_epoch))

    def obtain_all_label(self):
        logger = get_logger('basicda', self.logdir, logging.INFO)

        all_output = self.weak_score_bank
        all_fea = self.weak_feat_bank
        all_label = self.label_bank
        #
        _, predict = torch.max(all_output, 1)

        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        logger.info('orig acc is {}'.format(accuracy))
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

        all_fea = all_fea.float().cpu().numpy()
        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        #
        predict = predict.cpu().numpy()
        for _ in range(2):
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            cls_count = np.eye(K)[predict].sum(axis=0)
            labelset = np.where(cls_count > self.threshold)
            labelset = labelset[0]

            dd = cdist(all_fea, initc[labelset], 'cosine')
            pred_label = dd.argmin(axis=1)
            predict = labelset[pred_label]

            aff = np.eye(K)[predict]

        acc = np.sum(predict == all_label.cpu().float().numpy()) / len(all_fea)
        logger.info('acc is {}'.format(acc))
        prototype = torch.from_numpy(initc).to("cuda:{}".format(self.local_rank)).to(torch.float32)
        self.class_prototype_bank = F.normalize(prototype)

    def update_source_bank(self):
        base_model = self.source_model_dict['base_model']
        base_model.eval()
        with torch.no_grad():
            for data in self.train_loaders[0]:
                img = data[0]['img']
                img_ind = data[0]['image_ind']

                tmp_res = base_model(img.to('cuda:{}'.format(self.local_rank)))
                feat, logits, _, _ = tmp_res

                tmp_feat = feat
                tmp_score = F.softmax(logits, dim=-1)

                if self.is_distributed:
                    feat = concat_all_gather(tmp_feat)
                    score = concat_all_gather(tmp_score)
                    img_ind = concat_all_gather(img_ind.to('cuda:{}'.format(self.local_rank)))
                else:
                    feat = tmp_feat.detach()
                    score = tmp_score.detach()
                    img_ind = img_ind.detach()
                self.source_aad_weak_feat_bank[img_ind] = F.normalize(feat, dim=-1)
                self.source_aad_weak_score_bank[img_ind] = score

    def update_bank(self):
        self.set_eval_state()
        base_model = self.model_dict['base_model']
        shape = 0
        with torch.no_grad():
            for data in self.train_loaders[0]:
                img = data[0]['img']
                img_ind = data[0]['image_ind']
                img_label = data[0]['gt_label']

                tmp_res = base_model(img.to('cuda:{}'.format(self.local_rank)))
                feat, logits, target_feat, target_logits = tmp_res

                tmp_feat = feat
                tmp_score = F.softmax(logits, dim=-1)

                if self.is_distributed:
                    feat = concat_all_gather(tmp_feat)
                    score = concat_all_gather(tmp_score)
                    img_ind = concat_all_gather(img_ind.to('cuda:{}'.format(self.local_rank)))
                    img_label = concat_all_gather(img_label.to('cuda:{}'.format(self.local_rank)))
                else:
                    feat = tmp_feat.detach()
                    score = tmp_score.detach()
                    img_ind = img_ind.detach()
                    img_label = img_label.detach()
                self.weak_feat_bank[img_ind] = F.normalize(feat, dim=-1)
                self.weak_score_bank[img_ind] = score
                self.label_bank[img_ind] = img_label.squeeze(1).to('cuda:{}'.format(self.local_rank))

                if self.iteration == 0:
                    if self.is_distributed:
                        target_feat = concat_all_gather(target_feat)
                        target_score = concat_all_gather(F.softmax(target_logits, dim=-1))
                    else:
                        target_feat = target_feat
                        target_score = F.softmax(target_logits, dim=-1)
                    self.aad_weak_feat_bank[img_ind] = F.normalize(target_feat, dim=-1)
                    self.aad_weak_score_bank[img_ind] = target_score
                shape += img.shape[0]
        print('rank {}, shape {}'.format(self.local_rank, shape))
        self.set_train_state()

    def obtain_sim_mat(self):
        base_model = self.model_dict['base_model']
        if self.is_distributed:
            fc_weight = base_model.module.online_classifier.fc.weight_v.detach()
        else:
            fc_weight = base_model.online_classifier.fc.weight_v.detach()
        normalized_fc_weight = F.normalize(fc_weight)
        sim_mat_orig = normalized_fc_weight @ normalized_fc_weight.T
        eye_mat = torch.eye(self.num_class).to("cuda:{}".format(self.local_rank))
        non_eye_mat = 1 - eye_mat
        sim_mat = (eye_mat + non_eye_mat * sim_mat_orig * self.non_diag_alpha).clone()
        return sim_mat

    def step_grad_all(self):
        for name in self.optimizer_dict.keys():
            self.optimizer_dict[name].step()

    def zero_grad_all(self):
        for name in self.optimizer_dict.keys():
            self.optimizer_dict[name].zero_grad()

    def update_source_weak_bank_timely(self, feat, score, ind):
        with torch.no_grad():
            single_output_f_ = F.normalize(feat).detach().clone()
            tmp_softmax_out = score
            tmp_img_ind = ind
            if self.is_distributed:
                output_f_ = concat_all_gather(single_output_f_)
                tmp_softmax_out = concat_all_gather(tmp_softmax_out)
                tmp_img_ind = concat_all_gather(tmp_img_ind)
            else:
                output_f_ = single_output_f_
                tmp_softmax_out = tmp_softmax_out
                tmp_img_ind = tmp_img_ind

            self.source_aad_weak_feat_bank[tmp_img_ind] = output_f_.detach().clone()
            self.source_aad_weak_score_bank[tmp_img_ind] = tmp_softmax_out.detach().clone()

    def update_weak_bank_timely(self, feat, score, ind):
        with torch.no_grad():
            single_output_f_ = F.normalize(feat).detach().clone()
            tmp_softmax_out = score
            tmp_img_ind = ind
            if self.is_distributed:
                output_f_ = concat_all_gather(single_output_f_)
                tmp_softmax_out = concat_all_gather(tmp_softmax_out)
                tmp_img_ind = concat_all_gather(tmp_img_ind)
            else:
                output_f_ = single_output_f_
                tmp_softmax_out = tmp_softmax_out
                tmp_img_ind = tmp_img_ind

            self.aad_weak_feat_bank[tmp_img_ind] = output_f_.detach().clone()
            self.aad_weak_score_bank[tmp_img_ind] = tmp_softmax_out.detach().clone()

    def train_iter(self, *args):
        tgt_unlabeled_img_weak = args[0][0]['img'].squeeze(0)
        tgt_unlabeled_img_strong = args[0][1]['img'].squeeze(0)
        tgt_unlabeled_img_strong_2 = args[0][2]['img'].squeeze(0)
        tgt_img_ind = args[0][0]['image_ind']
        tgt_unlabeled_label = args[0][0]['gt_label'].squeeze(0)

        tgt_unlabeled_size = tgt_unlabeled_img_weak.shape[0]

        batch_metrics = {}
        batch_metrics['loss'] = {}

        base_model = self.model_dict['base_model']
        if self.iteration % self.pseudo_update_interval == 0:
            self.update_bank()
            self.update_source_bank()
            self.obtain_all_label()

        if self.iteration == 0:
            self.class_contrastive_simmat = self.obtain_sim_mat()
            self.instance_contrastive_simmat = self.obtain_sim_mat()

        with torch.cuda.amp.autocast():
            all_weak_img = tgt_unlabeled_img_weak
            all_strong_img = torch.cat((tgt_unlabeled_img_strong, tgt_unlabeled_img_strong_2), dim=0)
            tmp_res = base_model(all_strong_img, all_weak_img, src_labeled_size=0,
                                 tgt_unlabeled_size=tgt_unlabeled_size,
                                 train_iter=self.iteration)
            online_output, target_output, _, _ = tmp_res
            online_strong_logits = online_output['strong_logits']
            target_strong_logits = target_output['strong_logits']
            target_weak_logits = target_output['weak_logits']
            target_weak_feat = target_output['weak_feat']
            online_weak_logits = online_output['weak_logits']
            online_weak_feat = online_output['weak_feat']
            online_strong_prob = F.softmax(online_strong_logits, dim=-1)
            target_strong_prob = F.softmax(target_strong_logits, dim=-1)
            online_weak_prob = F.softmax(online_weak_logits, dim=-1)
            target_weak_prob = F.softmax(target_weak_logits, dim=-1)

            source_model = self.source_model_dict["base_model"]
            source_model.eval()
            source_feat, source_logits, _, _ = source_model(torch.cat([all_weak_img, all_strong_img], dim=0))
            source_weak_feat = source_feat[:tgt_unlabeled_size]
            source_strong_feat = source_feat[tgt_unlabeled_size:]
            source_weak_logits = source_logits[:tgt_unlabeled_size]
            source_strong_logits = source_logits[tgt_unlabeled_size:]
            source_weak_prob = F.softmax(source_weak_logits, dim=-1)
            source_strong_prob = F.softmax(source_strong_logits, dim=-1)

            self.update_weak_bank_timely(target_weak_feat, target_weak_prob, tgt_img_ind)
            self.update_source_weak_bank_timely(source_weak_feat, source_weak_prob, tgt_img_ind)

            loss, loss_source = self.baseline_loss(online_weak_prob, target_weak_feat, online_weak_logits, online_strong_logits[0:tgt_unlabeled_size],
                                                   target_weak_prob,
                                                   source_weak_prob, source_weak_feat, source_weak_logits, source_strong_logits[0:tgt_unlabeled_size],
                                                   batch_metrics, tgt_unlabeled_label)

            # fixmatch损失
            pseudo_label = torch.softmax(target_weak_logits.detach(), dim=-1)
            max_probs, tgt_u = torch.max(pseudo_label, dim=-1)
            tgt_u = self.obtain_batch_label(online_weak_feat)

            mask = max_probs.ge(self.prob_threshold).float().detach()
            pred_right = torch.sum((tgt_u == tgt_unlabeled_label.squeeze(1)) * mask) / torch.sum(mask)

            max_probs_source, tgt_u_source = torch.max(source_weak_prob.detach(), dim=-1)

            if self.fusion_flag and not self.second_stage:
                mask = (max_probs.ge(self.prob_threshold) & max_probs_source.ge(self.prob_threshold)
                        & (tgt_u == tgt_u_source)).float().detach()
            else:
                mask = max_probs.ge(self.prob_threshold).float().detach()
            mask_source = max_probs_source.ge(self.prob_threshold).float().detach()

            cluster_acc = torch.sum((tgt_u == tgt_unlabeled_label.squeeze(1)) * mask) / torch.sum(mask)

            mask_val = torch.sum(mask).item() / mask.shape[0]

            loss_1 = self.class_contrastive_loss(online_strong_prob[0:tgt_unlabeled_size], tgt_u, mask)
            loss_2 = self.class_contrastive_loss(online_strong_prob[tgt_unlabeled_size:], tgt_u, mask)
            loss += (loss_1 + loss_2) * self.lambda_fixmatch * 0.5

            loss_source += (self.class_contrastive_loss(source_strong_prob[0:tgt_unlabeled_size], tgt_u_source, mask_source) +
                            self.class_contrastive_loss(source_strong_prob[tgt_unlabeled_size:], tgt_u_source, mask_source)) * 0.5

            # constrastive loss
            all_k_strong = target_strong_prob
            all_k_weak = target_weak_prob
            weak_feat_for_backbone = online_weak_prob
            k_weak_for_backbone = all_k_weak
            k_strong_for_backbone = all_k_strong[0:tgt_unlabeled_size]
            strong_feat_for_backbone = online_strong_prob[0:tgt_unlabeled_size]
            k_strong_2 = all_k_strong[tgt_unlabeled_size:]

            tmp_weak_negative_bank = self.weak_negative_bank
            tmp_strong_negative_bank = self.strong_negative_bank

            info_nce_loss_1 = self.instance_contrastive_loss(strong_feat_for_backbone, k_weak_for_backbone,
                                                             tmp_weak_negative_bank)
            info_nce_loss_3 = self.instance_contrastive_loss(strong_feat_for_backbone, k_strong_2,
                                                             tmp_strong_negative_bank)
            info_nce_loss_2 = self.instance_contrastive_loss(weak_feat_for_backbone, k_strong_for_backbone,
                                                             tmp_strong_negative_bank)
            info_nce_loss = (info_nce_loss_1 + info_nce_loss_2 + info_nce_loss_3) / 3.0

            loss += info_nce_loss * self.lambda_nce

        self.scaler.scale(loss).backward()
        self.scaler_source.scale(loss_source).backward()

        self.scaler.step(self.optimizer_dict["base_model"])
        self.scaler.update()

        self.scaler_source.step(self.source_optimizer_dict["base_model"])
        self.scaler_source.update()

        self.optimizer_dict["base_model"].zero_grad()
        self.source_optimizer_dict["base_model"].zero_grad()

        self.update_negative_bank(target_weak_prob, target_strong_prob[0:tgt_unlabeled_size, :])

        batch_metrics['loss']['info_nce'] = info_nce_loss.item() if isinstance(info_nce_loss,
                                                                               torch.Tensor) else info_nce_loss
        batch_metrics['loss']['mean_max_prob'] = torch.mean(max_probs).item()
        batch_metrics['loss']['source_mean_max_prob'] = torch.mean(max_probs_source).item()
        batch_metrics['loss']['mask'] = mask_val
        batch_metrics['loss']['mask_acc'] = pred_right.item()
        batch_metrics['loss']['cluster_mask_acc'] = cluster_acc.item()

        return batch_metrics

    def update_negative_bank(self, weak_score, strong_score):
        def update_bank(new_score, bank, ptr):
            if self.is_distributed:
                all_score = concat_all_gather(new_score)
            else:
                all_score = new_score
            batch_size = all_score.shape[0]
            start_point = int(ptr)
            end_point = min(start_point + batch_size, self.bank_size)

            bank[start_point:end_point, :] = all_score[0:(end_point - start_point), :]
            if end_point == self.bank_size:
                ptr[0] = 0
            else:
                ptr += batch_size

        update_bank(weak_score, self.weak_negative_bank, self.weak_negative_bank_ptr)
        update_bank(strong_score, self.strong_negative_bank, self.strong_negative_bank_ptr)

    def get_contrastive_labels(self, query_feat):
        current_batch_size = query_feat.shape[0]
        constrastive_labels = torch.zeros((current_batch_size,), dtype=torch.long,
                                          device='cuda:{}'.format(self.local_rank))
        return constrastive_labels

    def instance_contrastive_loss(self, query_feat, key_feat, neg_feat):
        pos_logits = self.my_sim_compute(query_feat, key_feat, self.instance_contrastive_simmat, expand=False) * 0.5
        neg_logits = self.my_sim_compute(query_feat, neg_feat, self.instance_contrastive_simmat, expand=True) * 0.5
        all_logits = torch.cat((pos_logits, neg_logits), dim=1) / self.lambda_temp
        #
        constrastive_labels = self.get_contrastive_labels(query_feat)
        info_nce_loss = F.cross_entropy(all_logits, constrastive_labels) * 0.5
        return info_nce_loss

    def class_contrastive_loss(self, score, label, mask):
        all_other_prob = torch.eye(self.num_class).to("cuda:{}".format(self.local_rank))
        new_logits = self.my_sim_compute(score, all_other_prob, self.class_contrastive_simmat,
                                         expand=True) / self.lambda_fixmatch_temp
        loss_consistency = (F.cross_entropy(new_logits, label, reduction='none') * mask).mean()
        return loss_consistency

    def my_sim_compute(self, prob_1, prob_2, sim_mat, expand=True):
        b1 = prob_1.shape[0]
        b2 = prob_2.shape[0]
        cls_num = prob_1.shape[1]
        if expand:
            prob_1 = prob_1.unsqueeze(2).unsqueeze(1).expand(-1, b2, -1, -1)  # B1xB2xCx1
            prob_2 = prob_2.unsqueeze(1).unsqueeze(0).expand(b1, -1, -1, -1)  # B1xB2x1xC
            prob_1 = prob_1.reshape(b1 * b2, cls_num, 1)
            prob_2 = prob_2.reshape(b1 * b2, 1, cls_num)
            sim = torch.sum(torch.sum(torch.bmm(prob_1, prob_2) * sim_mat, -1), -1)
            sim = sim.reshape(b1, b2)
        else:
            prob_1 = prob_1.unsqueeze(2)  # BxCx1
            prob_2 = prob_2.unsqueeze(1)  # Bx1xC
            sim = torch.sum(torch.sum(torch.bmm(prob_1, prob_2) * sim_mat, -1), -1)
            sim = sim.reshape(b1, 1)
        return sim

    def obtain_batch_label(self, feat):
        feat_1 = F.normalize(feat.detach())
        prototype = F.normalize(self.class_prototype_bank.detach())

        cos_similarity = torch.mm(feat_1, prototype.t())
        pred_label = torch.argmax(cos_similarity, dim=1)
        return pred_label

    def baseline_loss(self, score, feat, logits, logits_strong, score_ema, score_source, feat_source, logits_source, logits_strong_source, batch_metrics,
                      real_label):
        if self.use_centroid:
            loss_aad_pos, loss_aad_neg, loss_source = self.AaD_loss_centroid(score, feat, logits, logits_strong, score_ema,
                                                                             score_source,
                                                                             feat_source, logits_source, logits_strong_source, real_label,
                                                                             batch_metrics)
        else:
            loss_aad_pos, loss_aad_neg, loss_source = self.AaD_loss(score, feat, score_ema, score_source, feat_source,
                                                                    real_label, batch_metrics)
        batch_metrics['loss']['aad_pos'] = loss_aad_pos.item()
        batch_metrics['loss']['aad_neg'] = loss_aad_neg.item()
        tmp_lambda = (1 + 10 * self.iteration / self.max_iters) ** (-self.beta)
        return (loss_aad_pos + loss_aad_neg * tmp_lambda) * self.lambda_aad, loss_source

    def obtain_neighbor(self, feat, feat_bank):
        normalized_feat = F.normalize(feat, dim=-1)
        if self.is_distributed:
            normalized_feat = concat_all_gather(normalized_feat)

        rand_idxs = torch.randperm(len(feat_bank)).to("cuda:{}".format(self.local_rank))
        banks = torch.cat([normalized_feat, feat_bank[rand_idxs][:self.memory_size]], dim=0)

        distance = banks @ feat_bank.T
        core_distance, idx_near = torch.topk(distance,
                                             dim=-1,
                                             largest=True,
                                             sorted=True,
                                             k=self.num_k + 1)

        return core_distance[:, -1], idx_near[:normalized_feat.shape[0]], normalized_feat.shape[0]

    def gather(self, score):
        tensors_gather = [torch.ones_like(score) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, score, async_op=False)
        tensors_gather[self.local_rank] = score
        outputs = torch.cat(tensors_gather, dim=0)

        return outputs

    def AaD_loss(self, score, feat, score_ema, score_source, feat_source, real_label, batch_metrics):
        with torch.no_grad():
            core_distance, idx_near, use_shape = self.obtain_neighbor(feat, self.aad_weak_feat_bank)
            source_core_distance, source_idx_near, _ = self.obtain_neighbor(feat_source, self.source_aad_weak_feat_bank)
            idx_near = idx_near[:, 1:]
            source_idx_near = source_idx_near[:, 1:]

            if self.is_distributed:
                pred_class = torch.argmax(concat_all_gather(score_ema), dim=-1)
                source_pred_class = torch.argmax(concat_all_gather(score_source), dim=-1)
            else:
                pred_class = torch.argmax(score_ema, dim=-1)
                source_pred_class = torch.argmax(score_source, dim=-1)

            if self.fusion_flag:
                use_source_batch = (torch.argsort(torch.argsort(source_core_distance)) >=
                                    torch.argsort(torch.argsort(core_distance))).float()
                use_source_batch = use_source_batch[:use_shape]
                # record
                used_label = source_pred_class * use_source_batch + pred_class * (1. - use_source_batch)
                used_label = used_label.long()
                if self.is_distributed:
                    real_label = concat_all_gather(real_label)
                real_label = real_label.squeeze(1)
                used_acc = torch.mean((used_label == real_label).float())
                batch_metrics['loss']['paradox_acc'] = used_acc.item()
                batch_metrics['loss']['target_acc'] = torch.mean((pred_class == real_label).float()).item()
                batch_metrics['loss']['source_acc'] = torch.mean((source_pred_class == real_label).float()).item()
                batch_metrics['loss']['use_source'] = torch.mean(use_source_batch.float()).item()

                use_source_batch = use_source_batch.unsqueeze(1)
                idx_near_final = source_idx_near * use_source_batch + idx_near * (1 - use_source_batch)
                score_near = self.aad_weak_score_bank[idx_near_final.long()]
                source_score_near = self.source_aad_weak_score_bank[idx_near_final.long()]

                # record
                batch_metrics['loss']['paradox_nn_acc'] = torch.mean((self.label_bank[idx_near_final.long()] ==
                                                                      real_label.unsqueeze(1)).float()).item()
                batch_metrics['loss']['target_nn_acc'] = torch.mean((self.label_bank[idx_near] ==
                                                                     real_label.unsqueeze(1)).float()).item()
                batch_metrics['loss']['source_nn_acc'] = torch.mean((self.label_bank[source_idx_near] ==
                                                                     real_label.unsqueeze(1)).float()).item()

                score_near = score_near.detach()
                source_score_near = source_score_near.detach()
            else:
                # record
                if self.is_distributed:
                    real_label = concat_all_gather(real_label)
                real_label = real_label.squeeze(1)
                batch_metrics['loss']['paradox_acc'] = 0
                batch_metrics['loss']['target_acc'] = torch.mean((pred_class == real_label).float()).item()
                batch_metrics['loss']['source_acc'] = torch.mean((source_pred_class == real_label).float()).item()
                batch_metrics['loss']['paradox_nn_acc'] = 0
                batch_metrics['loss']['target_nn_acc'] = torch.mean((self.label_bank[idx_near] ==
                                                                     real_label.unsqueeze(1)).float()).item()
                batch_metrics['loss']['source_nn_acc'] = torch.mean((self.label_bank[source_idx_near] ==
                                                                     real_label.unsqueeze(1)).float()).item()
                batch_metrics['loss']['use_source'] = 0

                score_near = self.aad_weak_score_bank[idx_near]  # batch x K x C
                source_score_near = self.source_aad_weak_score_bank[source_idx_near]
        #
        if self.is_distributed:
            outputs = self.gather(score)
            outputs_source = self.gather(score_source)
        else:
            outputs = score
            outputs_source = score_source

        softmax_out_un = outputs.unsqueeze(1).expand(-1, self.num_k, -1)  # batch x K x C
        loss_1 = torch.mean((F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1)).sum(1))

        softmax_out_un_source = outputs_source.unsqueeze(1).expand(-1, self.num_k, -1)  # batch x K x C
        loss_source = torch.mean((F.kl_div(softmax_out_un_source, source_score_near, reduction='none').sum(-1)).sum(1))

        mask = torch.ones((idx_near.shape[0], idx_near.shape[0]))
        diag_num = torch.diag(mask)
        mask_diag = torch.diag_embed(diag_num)
        mask = mask - mask_diag

        copy = outputs.T
        dot_neg = outputs @ copy
        dot_neg = (dot_neg * mask.to("cuda:{}".format(self.local_rank))).sum(-1)
        neg_pred = torch.mean(dot_neg)

        dot_neg_source = outputs_source @ outputs_source.T
        dot_neg_source = (dot_neg_source * mask.to("cuda:{}".format(self.local_rank))).sum(-1)
        dot_neg_source = torch.mean(dot_neg_source)

        loss_source += dot_neg_source
        return loss_1, neg_pred, loss_source

    def origin_kl_loss(self, p, q, logits, logits_strong, confidence):
        # p: target, q: logit
        t = -1.0 * q * p + q * torch.sum(q * p, dim=-1, keepdim=True)
        t = t.detach()

        if self.second_stage:
            kl_loss = torch.mean((t * logits).sum(1))
        else:
            kl_loss = torch.mean((t * logits).sum(1) * confidence.detach())

        t2 = -1.0 * q + torch.nn.Softmax(-1)(logits_strong)
        t2 = t2.detach()

        kl_loss += self.kl_weight * torch.mean((t2 * logits_strong).sum(1))

        return kl_loss

    def AaD_loss_centroid(self, score, feat, logits, logits_strong, score_ema, score_source, feat_source, logits_source, logits_strong_source, real_label,
                          batch_metrics):
        with torch.no_grad():
            core_distance, idx_near, use_shape = self.obtain_neighbor(feat, self.aad_weak_feat_bank)
            source_core_distance, source_idx_near, _ = self.obtain_neighbor(feat_source, self.source_aad_weak_feat_bank)

            if self.is_distributed:
                pred_class = torch.argmax(concat_all_gather(score_ema), dim=-1)
                source_pred_class = torch.argmax(concat_all_gather(score_source), dim=-1)
            else:
                pred_class = torch.argmax(score_ema, dim=-1)
                source_pred_class = torch.argmax(score_source, dim=-1)

            if self.fusion_flag:
                use_source_batch = (torch.argsort(torch.argsort(source_core_distance)) >=
                                    torch.argsort(torch.argsort(core_distance))).float()
                use_source_batch = use_source_batch[:use_shape]
                # record
                used_label = source_pred_class * use_source_batch + pred_class * (1. - use_source_batch)
                used_label = used_label.long()
                if self.is_distributed:
                    real_label = concat_all_gather(real_label)
                real_label = real_label.squeeze(1)
                used_acc = torch.mean((used_label == real_label).float())
                batch_metrics['loss']['paradox_acc'] = used_acc.item()
                batch_metrics['loss']['target_acc'] = torch.mean((pred_class == real_label).float()).item()
                batch_metrics['loss']['source_acc'] = torch.mean((source_pred_class == real_label).float()).item()
                batch_metrics['loss']['use_source'] = torch.mean(use_source_batch.float()).item()

                use_source_batch = use_source_batch.unsqueeze(1)

                # concat logit directly
                score_near_logit = torch.mean(self.aad_weak_score_bank[idx_near], dim=1)
                source_score_near_logit = torch.mean(self.source_aad_weak_score_bank[source_idx_near], dim=1)
                logit_concat = source_score_near_logit * use_source_batch + score_near_logit * (1. - use_source_batch)

                # concat logit nn
                idx_near_final = source_idx_near * use_source_batch + idx_near * (1 - use_source_batch)
                score_near = torch.mean(self.aad_weak_score_bank[idx_near_final.long()], dim=1)
                source_score_near = torch.mean(self.source_aad_weak_score_bank[idx_near_final.long()], dim=1)

                # record
                batch_metrics['loss']['paradox_nn_acc'] = torch.mean((self.label_bank[idx_near_final.long()[:, 1:]] ==
                                                                      real_label.unsqueeze(1)).float()).item()
                batch_metrics['loss']['target_nn_acc'] = torch.mean((self.label_bank[idx_near[:, 1:]] ==
                                                                     real_label.unsqueeze(1)).float()).item()
                batch_metrics['loss']['source_nn_acc'] = torch.mean((self.label_bank[source_idx_near[:, 1:]] ==
                                                                     real_label.unsqueeze(1)).float()).item()

                # concat paradox
                paradox_mask = (pred_class != source_pred_class).float()
                paradox_mask = paradox_mask.unsqueeze(-1)

                if not self.second_stage:
                    score_near = logit_concat * paradox_mask + score_near * (1. - paradox_mask)
                    source_score_near = logit_concat * paradox_mask + source_score_near * (1. - paradox_mask)
                score_near = score_near.detach()
                source_score_near = source_score_near.detach()
            else:
                # record
                if self.is_distributed:
                    real_label = concat_all_gather(real_label)
                real_label = real_label.squeeze(1)
                batch_metrics['loss']['paradox_acc'] = 0
                batch_metrics['loss']['target_acc'] = torch.mean((pred_class == real_label).float()).item()
                batch_metrics['loss']['source_acc'] = torch.mean((source_pred_class == real_label).float()).item()
                batch_metrics['loss']['paradox_nn_acc'] = 0
                batch_metrics['loss']['target_nn_acc'] = torch.mean((self.label_bank[idx_near[:, 1:]] ==
                                                                     real_label.unsqueeze(1)).float()).item()
                batch_metrics['loss']['source_nn_acc'] = torch.mean((self.label_bank[source_idx_near[:, 1:]] ==
                                                                     real_label.unsqueeze(1)).float()).item()
                batch_metrics['loss']['use_source'] = 0

                score_near = self.aad_weak_score_bank[idx_near]  # batch x K x C
                source_score_near = self.source_aad_weak_score_bank[source_idx_near]
                score_near = torch.mean(score_near, dim=1)
                source_score_near = torch.mean(source_score_near, dim=1)
        #
        if self.is_distributed:
            outputs = self.gather(score)
            outputs_source = self.gather(score_source)

            logits_out = self.gather(logits)
            logits_source_out = self.gather(logits_source)

            logits_out_strong = self.gather(logits_strong)
            logits_source_out_strong = self.gather(logits_strong_source)
        else:
            outputs = score
            outputs_source = score_source

            logits_out = logits
            logits_source_out = logits_source

            logits_out_strong = logits_strong
            logits_source_out_strong = logits_strong_source

        centroid_target = torch.mean(self.aad_weak_score_bank[idx_near_final.long()], dim=1)
        centroid_source = torch.mean(self.source_aad_weak_score_bank[idx_near_final.long()], dim=1)
        kl = F.kl_div(centroid_source.log(), centroid_target, reduction="none").sum(-1)
        if self.is_distributed:
            confidence, _ = torch.max((concat_all_gather(score_ema) + concat_all_gather(score_source)) / 2, dim=-1)
        else:
            confidence, _ = torch.max((score_ema + score_source) / 2, dim=-1)
        confidence = 2 * confidence / (1 + confidence)

        batch_metrics['loss']['confidence'] = torch.mean(confidence).item()

        loss_1 = self.weight * self.origin_kl_loss(score_near, outputs, logits_out, logits_out_strong, confidence)
        loss_source = self.weight * self.origin_kl_loss(source_score_near, outputs_source, logits_source_out, logits_source_out_strong, confidence)

        mask = torch.ones((idx_near.shape[0], idx_near.shape[0]))
        diag_num = torch.diag(mask)
        mask_diag = torch.diag_embed(diag_num)
        mask = mask - mask_diag

        copy = outputs.T
        dot_neg = outputs @ copy
        dot_neg = (dot_neg * mask.to("cuda:{}".format(self.local_rank))).sum(-1)
        neg_pred = torch.mean(dot_neg)

        dot_neg_source = outputs_source @ outputs_source.T
        dot_neg_source = (dot_neg_source * mask.to("cuda:{}".format(self.local_rank))).sum(-1)
        dot_neg_source = torch.mean(dot_neg_source)

        loss_source += dot_neg_source
        return loss_1, neg_pred, loss_source

    def __call__(self, train_iteration=None):
        # 设置训练标志
        self.set_train_state()
        self.source_model_dict["base_model"].eval()

        logger = get_logger('basicda', self.logdir, logging.INFO)
        # 根据scheduler的记录设置迭代次数
        train_loader_num = len(self.train_loader_iterator)
        tmp_iteration = 0
        while tmp_iteration < train_iteration:
            if self.iteration == self.stop_iteration and self.fusion_flag:
                logger.info(f"Enter the second_stage At {self.stop_iteration} Iterations!!!")
                self.second_stage = True

            all_data = []
            for ind in range(train_loader_num):
                try:
                    all_data.append(next(self.train_loader_iterator[ind]))
                except StopIteration:
                    if self.is_distributed:
                        self.set_epoch(ind)
                    self.train_loader_iterator[ind] = self.train_loaders[ind].__iter__()
                    time.sleep(2)
                    all_data.append(next(self.train_loader_iterator[ind]))
            # 数据移动到GPU上
            relocated_data = move_data_to_gpu(all_data, self.local_rank)
            self.train_batch_output = self.train_iter(*relocated_data)

            self.metric.update(self.train_batch_output)
            if (self.iteration + 1) % 100 == 0:
                out = self.metric.gen_out()
                logger.info(
                    "Iteration {}---Mean_Max_Prob {}, Source_Mean_Max_Prob {}, Use_Source {}, Confidence {}\n"
                    "Mask {}, Mask_ACC {}, Cluster_Mask_ACC {}\n"
                    "Paradox_ACC {}, Target_Acc {}, Source_Acc {}\n"
                    "Paradox_NN_ACC {}, Target_NN_Acc {}, Source_NN_Acc {}\n".format(
                        self.iteration + 1, out["mean_max_prob"], out["source_mean_max_prob"], out["use_source"], out["confidence"],
                        out["mask"], out["mask_acc"], out["cluster_mask_acc"],
                        out["paradox_acc"], out["target_acc"], out["source_acc"],
                        out["paradox_nn_acc"], out["target_nn_acc"], out["source_nn_acc"]))
            #
            self.iteration += 1
            tmp_iteration += 1

    def load_pretrained_model(self, weights_path, is_distributed):
        weights = torch.load(weights_path, map_location='cpu')
        weights = weights['base_model']
        weights = OrderedDict((f"module.{k}" if not k.startswith("module.") else k, v) for k, v in weights.items())
        
        for key in weights:
            key_split = key.split('.')
            if key_split[1] in ['target_network', 'target_classifier']:
                key_split[1] = key_split[1].replace('target', 'online')
                online_key = '.'.join(key_split)
                weights[key] = weights[online_key]
        if not is_distributed:
            new_weights = OrderedDict()
            for key in weights:
                new_key = key.replace("module.", "")
                new_weights[new_key] = weights[key]
            self.model_dict['base_model'].load_state_dict(new_weights, strict=False)
        else:
            new_weights = OrderedDict()
            for key in weights:
                if "module." not in key:
                    new_key = "module." + key  # key.replace("module.", "")
                else:
                    new_key = key
                new_weights[new_key] = weights[key]
            resp = self.model_dict['base_model'].load_state_dict(new_weights, strict=False)


def deal_with_val_interval(val_interval, max_iters, trained_iteration=0):
    fine_grained_val_checkpoint = []

    def reduce_trained_iteration(val_checkpoint):
        new_val_checkpoint = []
        start_flag = False
        for tmp_checkpoint in val_checkpoint:
            if start_flag:
                new_val_checkpoint.append(tmp_checkpoint)
            else:
                if tmp_checkpoint >= trained_iteration:
                    if tmp_checkpoint > trained_iteration:
                        new_val_checkpoint.append(tmp_checkpoint)
                    start_flag = True
        return new_val_checkpoint

    if isinstance(val_interval, (int, float)):
        val_times = int(max_iters / val_interval)
        if val_times == 0:
            raise RuntimeError(
                'max_iters number {} should be larger than val_interval {}'.format(max_iters, val_interval))
        for i in range(1, val_times + 1):
            fine_grained_val_checkpoint.append(i * int(val_interval))
        if fine_grained_val_checkpoint[-1] != max_iters:
            fine_grained_val_checkpoint.append(max_iters)
        return reduce_trained_iteration(fine_grained_val_checkpoint)
    elif isinstance(val_interval, dict):
        current_checkpoint = 0
        milestone_list = sorted(val_interval.keys())
        assert milestone_list[0] > 0 and milestone_list[-1] <= max_iters, 'check val interval keys'
        # 如果最后一个不是max_iter，则按最后的interval计算
        if milestone_list[-1] != max_iters:
            val_interval[max_iters] = val_interval[milestone_list[-1]]
            milestone_list.append(max_iters)
        last_milestone = 0
        for milestone in milestone_list:
            tmp_interval = val_interval[milestone]
            tmp_val_times = int((milestone - last_milestone) / tmp_interval)
            for i in range(tmp_val_times):
                fine_grained_val_checkpoint.append(current_checkpoint + int(tmp_interval))
                current_checkpoint += int(tmp_interval)
            if fine_grained_val_checkpoint[-1] != milestone:
                fine_grained_val_checkpoint.append(milestone)
                current_checkpoint = milestone
            last_milestone = current_checkpoint
        return reduce_trained_iteration(fine_grained_val_checkpoint)
    else:
        raise RuntimeError('only single value or dict is acceptable for val interval')
