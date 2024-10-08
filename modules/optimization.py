import torch.nn.functional as F
import torch.nn as nn
import torch

class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()
        self.error_metric = nn.KLDivLoss(size_average=True, reduce=True)

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss
