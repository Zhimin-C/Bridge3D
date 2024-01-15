import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NCELoss(nn.Module):
    """
    Compute the PointInfoNCE loss
    """

    def __init__(self, temperature):
        super(NCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, k, q):
        logits = torch.mm(k, q.transpose(1, 0))
        target = torch.arange(k.shape[0], device=k.device).long()
        out = torch.div(logits, self.temperature)
        out = out.contiguous()

        loss = self.criterion(out, target)
        return loss

class semantic_NCELoss(nn.Module):
    """
    Compute the PointInfoNCE loss
    """

    def __init__(self, temperature):
        super(semantic_NCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, k, q, pseudo_label):
        logits = torch.mm(k, q.transpose(1, 0))
        # print(logits)
        target = torch.arange(k.shape[0], device=k.device).long()
        logits = torch.div(logits, self.temperature)
        # out = out.contiguous()

        permute = pseudo_label.unsqueeze(-1).repeat(1, pseudo_label.shape[0])
        mask = permute == permute.permute(1, 0)
        mask_diag = torch.diag_embed(torch.Tensor([True] * pseudo_label.shape[0])).to(k.device).bool()
        mask = mask & (~mask_diag)
        logits[mask] = 0
        logits_sparse = logits.to_sparse()
        logits_sparse = torch.sparse.log_softmax(logits_sparse, dim=1).to_dense()

        # d_sparse = d.to_sparse()
        # torch.sparse.log_softmax(d_sparse, dim=0)
        # torch.sparse.log_softmax(d_sparse, dim=1).to_dense()

        # import pdb
        # pdb.set_trace()
        loss = F.nll_loss(logits_sparse, target)

        # loss = self.criterion(out, target)
        return loss
