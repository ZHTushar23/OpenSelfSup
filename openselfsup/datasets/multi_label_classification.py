import torch
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
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

    def evaluate(self, scores, keyword, logger=None):
        eval_res = {}

        target = self.get_labels()
        assert scores.size(0) == target.size(0), \
            "Inconsistent length for results and labels, {} vs {}".format(
            scores.size(0), target.size(0))

        n_classes = scores.size(1)
        target = target.cpu().detach().numpy()
        scores = scores.cpu().detach().numpy()
        # For each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(target[:, i],
                                                                scores[:, i])
            average_precision[i] = average_precision_score(target[:, i], scores[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(target.ravel(),
                                                                        scores.ravel())
        average_precision["mAP"] = average_precision_score(target, scores,
                                                             average="micro")*100
        if logger is not None and logger != 'silent':
            print_log('Average precision score, '
                      'micro-averaged over all classes: {0:0.2f}'.format(average_precision["mAP"]),
                      logger=logger)

        return average_precision

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
        return label_tensor