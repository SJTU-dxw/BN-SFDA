import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models
from mmcv.runner import get_dist_info
import torch.nn.utils.weight_norm as weightNorm


resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        print('init batchnorm layer!!!!!!!!!!')
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        print('init linear layer!!!!!!!!!!!!')
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)
    for current_params, ma_params in zip(current_model.buffers(), ma_model.buffers()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


class SFDAResNetBase(nn.Module):
    def __init__(self, resnet_name, bottleneck_dim=256):
        super(SFDAResNetBase, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        #
        self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
        self.bn = nn.BatchNorm1d(bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.bottleneck_dim = bottleneck_dim

    def forward(self, x, normalize=False):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        x = self.bn(x)
        if normalize:
            x = F.normalize(x, dim=1)
        return x

    def output_num(self):
        return self.bottleneck_dim

    def optim_parameters(self, lr):
        parameter_list = [{"params": self.feature_layers.parameters(), "lr": lr},
                          {"params": self.bottleneck.parameters(), "lr": lr * 10},
                          {"params": self.bn.parameters(), 'lr': lr * 10},
                          ]
        return parameter_list


class SFDAClassifier(nn.Module):
    def __init__(self, num_class, bottleneck_dim=256, type="wn_linear"):
        super(SFDAClassifier, self).__init__()
        self.type = type
        if type == 'wn_linear':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, num_class), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, num_class)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

    def optim_parameters(self, lr):
        parameter_list = [{"params": self.fc.parameters(), "lr": lr}, ]
        return parameter_list


class BasicModel(nn.Module):
    def __init__(self, model_dict, classifier_dict):
        super(BasicModel, self).__init__()
        self.online_network = SFDAResNetBase(model_dict["resnet_name"], model_dict["bottleneck_dim"])
        self.online_classifier = SFDAClassifier(classifier_dict["num_class"], classifier_dict["bottleneck_dim"])

    def forward(self, x):
        feature_extractor = self.online_network
        classifier = self.online_classifier
        feat = feature_extractor(x)

        fc_weight = self.online_classifier.fc.weight_v.detach()
        fc_weight = F.normalize(fc_weight, dim=1)

        logits = F.normalize(feat) @ fc_weight.T
        logits /= 0.07
        # logits = classifier(feat)
        return feat, logits, None, None

    def optim_parameters(self, lr):
        params = []
        if hasattr(self.online_network, 'optim_parameters'):
            tmp_params = self.online_network.optim_parameters(lr)
        else:
            raise RuntimeError('not supported')
        params.extend(tmp_params)
        
        if hasattr(self.online_classifier, 'optim_parameters'):
            tmp_params = self.online_classifier.optim_parameters(lr * 10)
        else:
            raise RuntimeError('not supported')
        params.extend(tmp_params)
        return params


class BasicMixContrastiveModel(nn.Module):
    def __init__(self, model_dict, classifier_dict, num_class, low_dim=512,
                 model_moving_average_decay=0.99,
                 proto_moving_average_decay=0.99,
                 fusion_type='reconstruct_double_detach', normalize=True, all_normalize=False,
                 force_no_shuffle_bn=False, mixup_sample_type='low', select_src_by_tgt_similarity=False,
                 src_keep_ratio=0.5,
                 ):
        super(BasicMixContrastiveModel, self).__init__()
        self.fusion_type = fusion_type
        self.online_network = SFDAResNetBase(model_dict["resnet_name"], model_dict["bottleneck_dim"])
        self.target_network = SFDAResNetBase(model_dict["resnet_name"], model_dict["bottleneck_dim"])
        self.target_ema_updater = EMA(model_moving_average_decay)
        self.proto_moving_average_decay = proto_moving_average_decay
        self.normalize = normalize
        self.num_class = num_class
        self.low_dim = low_dim
        self.all_normalize = all_normalize
        self.force_no_shuffle_bn = force_no_shuffle_bn
        self.mixup_sample_type = mixup_sample_type
        self.select_src_by_tgt_similarity = select_src_by_tgt_similarity
        self.src_keep_ratio = src_keep_ratio
        rank, _ = get_dist_info()
        #
        self.online_classifier = SFDAClassifier(classifier_dict["num_class"], classifier_dict["bottleneck_dim"])
        self.target_classifier = SFDAClassifier(classifier_dict["num_class"], classifier_dict["bottleneck_dim"])
        self.gpu_device = 'cuda:{}'.format(rank)
        #
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.online_classifier.parameters(), self.target_classifier.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    # TODO: 模型输入需要考虑多个输入的情况
    def forward(self, x1, x2=None, src_mixup=False, **kwargs):
        threshold = kwargs.get('threshold', None)
        if 'threshold' in kwargs:
            kwargs.pop('threshold')
        src_labeled_size = kwargs.get('src_labeled_size', None)
        if 'src_labeled_size' in kwargs:
            kwargs.pop('src_labeled_size')
        tgt_labeled_size = kwargs.get('tgt_labeled_size', 0)  # 对于无监督域适应，这里的值为0
        if 'tgt_labeled_size' in kwargs:
            kwargs.pop('tgt_labeled_size')
        #
        x1_img = x1
        x2_img = x2
        if x2 is not None:
            all_labeled_size = src_labeled_size + tgt_labeled_size
            with torch.no_grad():
                # TODO:挪到后面去
                self.update_moving_average()
                #
                tmp_shuffle_bn_flag = True if not self.force_no_shuffle_bn else False
                target_res = self.model_forward(x1_img, x2_img, model_type='target', shuffle_bn=tmp_shuffle_bn_flag,
                                                **kwargs)
            mix_feat_list = None
            lambda_mixup = None
            #
            online_res = self.model_forward(x1_img.clone(), x2_img.clone(), model_type='online', **kwargs)
            #
            return online_res, target_res, lambda_mixup, mix_feat_list
        else:
            return self.single_input_forward(x1_img, **kwargs)

    def update_moving_average(self):
        update_moving_average(self.target_ema_updater, self.target_network, self.online_network)
        update_moving_average(self.target_ema_updater, self.target_classifier, self.online_classifier)

    def optim_parameters(self, lr):
        params = []
        if hasattr(self.online_network, 'optim_parameters'):
            tmp_params = self.online_network.optim_parameters(lr)
        else:
            raise RuntimeError('not supported')
        params.extend(tmp_params)
        #
        if hasattr(self.online_classifier, 'optim_parameters'):
            tmp_params = self.online_classifier.optim_parameters(lr * 10)
        else:
            raise RuntimeError('not supported')
        params.extend(tmp_params)
        return params

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]

        try:
            x_gather = concat_all_gather(x)
        except Exception as e:
            x_gather = x

        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all, device=self.gpu_device)
        # broadcast to all gpus
        try:
            torch.distributed.broadcast(idx_shuffle, src=0)
        except Exception as e:
            idx_shuffle = idx_shuffle

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        try:
            gpu_idx = torch.distributed.get_rank()
        except Exception as e:
            gpu_idx = 0

        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        try:
            x_gather = concat_all_gather(x)
        except Exception as e:
            x_gather = x
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        try:
            gpu_idx = torch.distributed.get_rank()
        except Exception as e:
            gpu_idx = 0
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this]

    def feat_prob_fusion(self, feat, prob, model='online'):
        if self.fusion_type == 'prob':
            return prob
        elif self.fusion_type == 'feat':
            return feat
        elif self.fusion_type == 'outer_product':
            return self.random_layer((feat, prob))
        elif self.fusion_type == 'reconstruct':
            if model == 'online':
                fc_weight = self.online_classifier.fc2.weight
            else:
                fc_weight = self.target_classifier.fc2.weight
            return prob.mm(fc_weight)
        elif self.fusion_type == 'reconstruct_detach':
            if model == 'online':
                fc_weight = self.online_classifier.fc2.weight.detach()
            else:
                fc_weight = self.target_classifier.fc2.weight.detach()
            fc_weight = F.normalize(fc_weight, dim=1) if (not self.normalize) or self.all_normalize else fc_weight
            return prob.mm(fc_weight)
        elif self.fusion_type == 'reconstruct_double_detach':
            if model == 'online':
                fc_weight = self.online_classifier.fc2.weight.detach()
            else:
                fc_weight = self.target_classifier.fc2.weight.detach()
            fc_weight = F.normalize(fc_weight, dim=1) if (not self.normalize) or self.all_normalize else fc_weight
            #
            new_prob = F.softmax(feat.mm(fc_weight.detach().t()) / self.online_classifier.temp, dim=1)
            return new_prob.mm(fc_weight)
        elif self.fusion_type == 'l1_reconstruct_detach':
            if model == 'online':
                fc_weight = self.online_classifier.fc2.weight
            else:
                fc_weight = self.target_classifier.fc2.weight
            # 计算系数
            coeff = feat.mm(fc_weight.t())
            coeff = coeff / torch.sum(coeff, dim=1, keepdim=True)
            return coeff.mm(fc_weight.detach())
        else:
            raise RuntimeError("need fusion_type")

    def test_forward(self, x1_img, **kwargs):
        feat = self.online_network(x1_img, **kwargs)
        logits = self.online_classifier(feat)
        prob = F.softmax(logits, dim=-1)
        fusion_feat = self.feat_prob_fusion(feat, prob, model='online')
        online_pred = self.online_classifier(feat)
        target_feat = self.target_network(x1_img, **kwargs)
        target_pred = self.target_classifier(target_feat)
        return feat, fusion_feat, online_pred, target_pred

    def model_forward(self, x1_img, x2_img, model_type=None, shuffle_bn=False, **kwargs):
        if model_type == "online":
            feature_extractor = self.online_network
            classifier = self.online_classifier
        elif model_type == 'target':
            feature_extractor = self.target_network
            classifier = self.target_classifier
        else:
            raise RuntimeError('wrong model type specified')
        x1_shape = x1_img.shape[0]
        img_concat = torch.cat((x1_img, x2_img))
        if shuffle_bn:
            img_concat, idx_unshuffle = self._batch_shuffle_ddp(img_concat)
        feat = feature_extractor(img_concat)
        if shuffle_bn:
            feat = self._batch_unshuffle_ddp(feat, idx_unshuffle)
        logits = classifier(feat)
        prob = F.softmax(logits, dim=-1)
        contrastive_feat = self.feat_prob_fusion(feat, prob, model=model_type)
        contrastive_feat = F.normalize(contrastive_feat, dim=1) if self.normalize else contrastive_feat
        #
        strong_logits = logits[0:x1_shape]
        weak_logits = logits[x1_shape:]
        strong_prob = prob[0:x1_shape]
        weak_prob = prob[x1_shape:]
        strong_contrastive_feat = contrastive_feat[0:x1_shape]
        weak_contrastive_feat = contrastive_feat[x1_shape:]
        output = {
            'strong_logits': strong_logits,
            'weak_logits': weak_logits,
            'strong_prob': strong_prob,
            'weak_prob': weak_prob,
            'strong_contrastive_feat': strong_contrastive_feat,
            'weak_contrastive_feat': weak_contrastive_feat,
        }
        return output

    def single_input_forward(self, x1_img, **kwargs):
        if self.training:
            feat = self.online_network(x1_img, **kwargs)
            pred = self.online_classifier(feat, reverse=True, eta=1.0)
            return pred
        else:
            res = self.test_forward(x1_img, **kwargs)
            return res


class SFDASimplifiedContrastiveModel(BasicMixContrastiveModel):
    def __init__(self, model_dict, classifier_dict, num_class, low_dim=128,
                 model_moving_average_decay=0.99,
                 proto_moving_average_decay=0.99,
                 fusion_type='reconstruct_double_detach',
                 force_no_shuffle_bn=False, forward_twice=False,
                 ):
        super(SFDASimplifiedContrastiveModel, self).__init__(model_dict, classifier_dict, num_class, low_dim=low_dim,
                                                             model_moving_average_decay=model_moving_average_decay,
                                                             proto_moving_average_decay=proto_moving_average_decay,
                                                             fusion_type=fusion_type, normalize=True,
                                                             force_no_shuffle_bn=force_no_shuffle_bn,
                                                             )
        rank, world_size = get_dist_info()
        self.local_rank = rank
        self.world_size = world_size

        self.forward_twice = forward_twice

    def test_forward(self, x1_img, **kwargs):
        feat = self.online_network(x1_img, **kwargs)
        online_pred = self.online_classifier(feat)
        target_feat = self.target_network(x1_img, **kwargs)
        target_pred = self.target_classifier(target_feat)
        return feat, online_pred, target_feat, target_pred

    def model_forward(self, x1_img, x2_img, model_type=None, shuffle_bn=False, **kwargs):
        if model_type == "online":
            feature_extractor = self.online_network
            classifier = self.online_classifier
        elif model_type == 'target':
            feature_extractor = self.target_network
            classifier = self.target_classifier
        else:
            raise RuntimeError('wrong model type specified')
        #
        x1_shape = x1_img.shape[0]
        img_concat = torch.cat((x1_img, x2_img))
        if shuffle_bn:
            img_concat, idx_unshuffle = self._batch_shuffle_ddp(img_concat)
        feat = feature_extractor(img_concat)
        if shuffle_bn:
            feat = self._batch_unshuffle_ddp(feat, idx_unshuffle)
        logits = classifier(feat)
        prob = F.softmax(logits, dim=-1)
        #
        strong_feat = feat[0:x1_shape]
        weak_feat = feat[x1_shape:]
        strong_logits = logits[0:x1_shape]
        weak_logits = logits[x1_shape:]
        strong_prob = prob[0:x1_shape]
        weak_prob = prob[x1_shape:]
        #
        output = {
            'strong_feat': strong_feat,
            'weak_feat': weak_feat,
            'strong_logits': strong_logits,
            'weak_logits': weak_logits,
            'strong_prob': strong_prob,
            'weak_prob': weak_prob,
        }
        return output

    def single_input_forward(self, x1_img, **kwargs):
        if self.training:
            feature_extractor = self.online_network
            classifier = self.online_classifier
            feat = feature_extractor(x1_img)
            logits = classifier(feat)
            output = {
                'feat': feat,
                'logits': logits,
            }
            return output
        else:
            return self.test_forward(x1_img, **kwargs)
