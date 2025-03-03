import torch
import torch.nn as nn
from torch.nn import Transformer

class FeatureTransformer(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, dim_feedforward):
        super(FeatureTransformer, self).__init__()
        self.transformer = Transformer(d_model=input_dim, nhead=num_heads, num_encoder_layers=num_layers, dim_feedforward=dim_feedforward)

    def forward(self, src):
        return self.transformer(src, src)