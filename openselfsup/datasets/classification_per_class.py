import torch
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from openselfsup.utils import print_log

from .registry import DATASETS
from .base import BaseDataset
from .utils import to_numpy


@DATASETS.register_module
class ClassificationPerClassDataset(BaseDataset):
    """Dataset for classification.
    """

    def __init__(self, data_source, pipeline, prefetch=False):
        super(ClassificationPerClassDataset, self).__init__(data_source, pipeline, prefetch)

    def __getitem__(self, idx):
        img, target = self.data_source.get_sample(idx)
        img = self.pipeline(img)
        if self.prefetch:
            img = torch.from_numpy(to_numpy(img))
        return dict(img=img, gt_label=target)

    def evaluate(self, scores, keyword, logger=None, topk=(1, 5)):
        eval_res = {}

        target = torch.LongTensor(self.data_source.labels)
        print("#############################################################")
        print("Target Shape", target.shape)
        assert scores.size(0) == target.size(0), \
            "Inconsistent length for results and labels, {} vs {}".format(
            scores.size(0), target.size(0))
        num = scores.size(0)
        _, pred = scores.topk(max(topk), dim=1, largest=True, sorted=True)
        pred=torch.squeeze(pred,1)
        target_names = ['Others', 'Melanoma']
        results = (classification_report(target, pred, target_names=target_names, digits=4))
        if logger is not None and logger != 'silent':
            print_log("\n"+results,
                logger=logger)
        eval_res=results
        return eval_res

    def get_labels(self):
        return torch.LongTensor(self.data_source.labels)
