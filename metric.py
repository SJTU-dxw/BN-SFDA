from collections import defaultdict


class Metric(object):
    def __init__(self):
        self.iteration = 0
        self.metric_dic = defaultdict(float)

    def update(self, batch_output):
        self.iteration += 1
        for k, v in batch_output['loss'].items():
            self.metric_dic[k] += v

    def gen_out(self):
        for k, v in self.metric_dic.items():
            self.metric_dic[k] = v / 100.0
        out_dic = self.metric_dic
        self.iteration = 0
        self.metric_dic = defaultdict(float)
        return out_dic
