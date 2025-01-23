from mmcls.datasets import BaseDataset
import os
import copy
from pipelines import Compose
import numpy as np
import random


def get_worker_init_fn(worker_id, num_workers, rank, seed):
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataset_fromlist(image_list):
    with open(image_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
    with open(image_list) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
    return image_index, label_list


class SSDA_TEST_Datasets(BaseDataset):
    def __init__(self, dataset_params):
        name_split = dataset_params["name"].split('_')
        task = name_split[0]
        dataset = name_split[1]
        img_root = os.path.join("data", task)
        image_list = os.path.join('data/txt', task, dataset_params["split"] + '_images_' +
                                  dataset + "_" + str(dataset_params["shot"]) + '.txt')
        self.min_len = 0
        self.ann_file = image_list
        super(SSDA_TEST_Datasets, self).__init__(data_prefix=img_root, pipeline=dataset_params["pipeline"],
                                                 ann_file=image_list)
        self.name = dataset_params["name"]
        self.split = dataset_params["split"]

    def load_annotations(self):
        domain_labels = []
        imgs, labels = make_dataset_fromlist(self.ann_file)
        domain_labels.extend([0] * len(labels))

        self.n_classes = max(labels) + 1
        repeat_num = int(float(self.min_len) / len(imgs)) + 1
        imgs = imgs * repeat_num
        labels = labels * repeat_num
        domain_labels = domain_labels * repeat_num  #
        data_infos = []
        for ind, (img, label, domain_label) in enumerate(zip(imgs, labels, domain_labels)):
            info = {'img_prefix': self.data_prefix, 'img_info': {'filename': img}, 'gt_label': label,
                    'domain_label': domain_label, 'image_ind': ind}
            data_infos.append(info)
        return data_infos

    def get_classes(self, classes=None):
        class_names = []
        if isinstance(self.ann_file, list):
            tmp_ann_file = self.ann_file[0]
        else:
            tmp_ann_file = self.ann_file
        with open(tmp_ann_file, 'rb') as f:
            for line in f.readlines():
                line = line.decode()
                tmp_res = line.strip().split(' ')
                tmp_path = tmp_res[0]
                # print(tmp_res[0],tmp_res[1])
                tmp_ind = int(tmp_res[1])
                tmp_name = tmp_path.split('/')[1]
                if tmp_ind == len(class_names):
                    class_names.append(tmp_name)
        return class_names


class SSDA_CLS_Datasets(BaseDataset):
    def __init__(self, dataset_params):
        name_split = dataset_params["name"].split('_')
        task = name_split[0]
        dataset = name_split[1]
        img_root = os.path.join("data", task)
        image_list = os.path.join('data/txt', task, dataset_params["split"] + '_images_' +
                                  dataset + "_" + str(dataset_params["shot"]) + '.txt')
        self.min_len = 0
        self.ann_file = image_list
        super(SSDA_CLS_Datasets, self).__init__(data_prefix=img_root, pipeline=dataset_params["pipeline"],
                                                ann_file=image_list)
        self.name = dataset_params["name"]
        self.split = dataset_params["split"]

        self.pipeline_2 = Compose(dataset_params["pipeline2"])
        self.repeat_pipeline = 2

    def __getitem__(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        aug_data_1 = self.pipeline(results)
        aug_data_2 = self.pipeline_2(results)
        aug_data_3 = self.pipeline_2(results)

        return aug_data_1, aug_data_2, aug_data_3

    def load_annotations(self):
        domain_labels = []
        imgs, labels = make_dataset_fromlist(self.ann_file)
        domain_labels.extend([0] * len(labels))

        self.n_classes = max(labels) + 1
        repeat_num = int(float(self.min_len) / len(imgs)) + 1
        imgs = imgs * repeat_num
        labels = labels * repeat_num
        domain_labels = domain_labels * repeat_num  #
        data_infos = []
        for ind, (img, label, domain_label) in enumerate(zip(imgs, labels, domain_labels)):
            info = {'img_prefix': self.data_prefix, 'img_info': {'filename': img}, 'gt_label': label,
                    'domain_label': domain_label, 'image_ind': ind}
            data_infos.append(info)
        return data_infos

    def get_classes(self, classes=None):
        class_names = []
        if isinstance(self.ann_file, list):
            tmp_ann_file = self.ann_file[0]
        else:
            tmp_ann_file = self.ann_file
        with open(tmp_ann_file, 'rb') as f:
            for line in f.readlines():
                line = line.decode()
                tmp_res = line.strip().split(' ')
                tmp_path = tmp_res[0]
                tmp_ind = int(tmp_res[1])
                tmp_name = tmp_path.split('/')[1]
                if tmp_ind == len(class_names):
                    class_names.append(tmp_name)
        return class_names
