import torch
from sklearn.metrics import balanced_accuracy_score

from openselfsup.utils import print_log

from .registry import DATASETS
from .base import BaseDataset
from .utils import to_numpy


@DATASETS.register_module
class ClassificationDataset(BaseDataset):
    """Dataset for classification.
    """

    def __init__(self, data_source, pipeline, prefetch=False):
        super(ClassificationDataset, self).__init__(data_source, pipeline, prefetch)

    def __getitem__(self, idx):
        img, target = self.data_source.get_sample(idx)
        img = self.pipeline(img)
        if self.prefetch:
            img = torch.from_numpy(to_numpy(img))
        return dict(img=img, gt_label=target)

    def evaluate(self, scores, keyword, logger=None, topk=(1, 5)):
        eval_res = {}

        target = torch.LongTensor(self.data_source.labels)
        assert scores.size(0) == target.size(0), \
            "Inconsistent length for results and labels, {} vs {}".format(
            scores.size(0), target.size(0))
        num = scores.size(0)
        _, pred = scores.topk(max(topk), dim=1, largest=True, sorted=True)

        ####################################################################
        _pred = torch.squeeze(pred,1)
        b_acc = balanced_accuracy_score(target, _pred)*100.0

        #######################################################################
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))  # KxN

        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0).item()
            acc = correct_k * 100.0 / num
            eval_res["{}_top{}".format(keyword, k)] = acc

            # if logger is not None and logger != 'silent':
            #     print_log(
            #         "{}_top{}: {:.03f}".format(keyword, k, acc),
            #         logger=logger)
            ########################################
            if logger is not None and logger != 'silent':
                print_log(
                    "balanced_acc: {:.03f}".format(b_acc),
                    logger=logger)
            #######################################

        return eval_res

    def get_labels(self):
        return torch.LongTensor(self.data_source.labels)
