# Inspired from https://github.com/vladisai/PLDM

from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import functional as F

@dataclass
class VCRegObjectiveConfig():
    std_coeff: float = 1.0
    cov_coeff: float = 0.04
    cov_per_feature: bool = False
    adjust_cov: bool = True
    cov_chunk_size: Optional[int] = None
    std_margin: float = 1


class VCRegObjective(torch.nn.Module):
    def __init__(
        self,
        config: VCRegObjectiveConfig
    ):
        super().__init__()
        self.config = config

    def __call__(self, z):
        loss_components = {}

        z_flat = z.view(z.shape[0], -1)  # Flatten the tensor to (B, D) shape
        std_loss = self.std_loss(z_flat)
        cov_loss = self.cov_loss(z_flat)

        loss = (
            + self.config.cov_coeff * cov_loss.mean()
            + self.config.std_coeff * std_loss.mean()
        )
        loss_components["cov_loss"] = cov_loss.mean()
        loss_components["std_loss"] = std_loss.mean()
        loss_components["loss"] = loss
        return loss, loss_components

    def std_loss(self, x: torch.Tensor):
        x = x - x.mean(dim=1, keepdim=True)  # mean for each dim across batch samples

        if self.config.std_coeff:
            std = torch.sqrt(x.var(dim=1) + 0.0001)
            std_margin = 1.0
            std_loss = torch.mean(F.relu(std_margin - std), dim=-1)
        else:
            std_loss = torch.zeros([1])

        return std_loss

    def cov_loss(self, x: torch.Tensor):
        batch_size = x.shape[0]
        num_features = x.shape[-1]

        x = x - x.mean(dim=1, keepdim=True)

        if self.config.cov_coeff:
            cov = torch.mm(x.transpose(0, 1), x) / (batch_size - 1)
            cov_loss = (cov.pow(2).sum() - cov.diag().pow(2).sum()) / num_features
            if self.config.adjust_cov:
                cov_loss = cov_loss / (
                    num_features - 1
                )  # divide by num of elements on off-diagonal.
                # in orig paper they divide by num_features
                # but the correct version is (num_features - 1)*num_features
        else:
            cov_loss = torch.zeros([1])

        return cov_loss
