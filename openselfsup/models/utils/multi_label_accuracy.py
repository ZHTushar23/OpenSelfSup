import torch.nn as nn
import torch
from sklearn.metrics import average_precision_score

def multi_label_accuracy(pred, target):

    target = target.cpu().detach().numpy()
    pred = pred.cpu().detach().numpy()
    average_precision = average_precision_score(target, pred,
                                                         average="micro")*100

    return torch.tensor(average_precision).cuda()


class MultiLabelAccuracy(nn.Module):

    def __init__(self, topk=(1, )):
        super().__init__()
        self.topk = topk

    def forward(self, pred, target):
        return multi_label_accuracy(pred, target)
