import torch

from openselfsup.utils import print_log

from .registry import DATASETS
from .base import BaseDataset
from .utils import to_numpy


@DATASETS.register_module
class MultiLabelClassificationDataset(BaseDataset):
    """Dataset for classification.
    """

    def __init__(self, data_source, pipeline, prefetch=False):
        super(MultiLabelClassificationDataset, self).__init__(data_source, pipeline, prefetch)

    def __getitem__(self, idx):
        img, target = self.data_source.get_sample(idx)
        target = self.hot_encode(target)
        img = self.pipeline(img)
        if self.prefetch:
            img = torch.from_numpy(to_numpy(img))
        return dict(img=img, gt_label=target)

    def evaluate(self, scores, keyword, logger=None, topk=(1, 5)):
        eval_res = {}

        target = self.get_labels()
        assert scores.size(0) == target.size(0), \
            "Inconsistent length for results and labels, {} vs {}".format(
            scores.size(0), target.size(0))
        num = scores.size(0)
        _, pred = scores.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        print("#################################################################################################")
        print(target.shape)
        print(scores.shape)
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # KxN

        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0).item()
            acc = correct_k * 100.0 / num
            eval_res["{}_top{}".format(keyword, k)] = acc

            if logger is not None and logger != 'silent':
                print_log(
                    "{}_top{}: {:.03f}".format(keyword, k, acc),
                    logger=logger)

        return eval_res

    def get_labels(self):
        labels = self.data_source.labels
        #empty tensor to store labels with hot encoding
        tensor_labels = torch.zeros(len(labels),self.data_source.total_classes)
        tensor_labels.long()
        for idx in range(len(labels)):
            tensor_labels[idx,:] = self.hot_encode(labels[idx])

        return tensor_labels

    def hot_encode(self,label):
        label_tensor = torch.zeros(self.data_source.total_classes)
        idx = [int(n) for n in label]
        label_tensor[idx] =1
        return label_tensor.long()