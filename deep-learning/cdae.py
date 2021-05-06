"""
Yao Wu et al., Collaborative denoising auto-encoders for top-n recommender systems. WSDM 2016.
https://alicezheng.org/papers/wsdm16-cdae.pdf
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from BaseModel import BaseModel

class CDAE(BaseModel):
    """
    Collaborative Denoising Autoencoder model class
    """
    def __init__(self, model_conf, num_users, num_items, device):
        """
        :param model_conf: model configuration
        :param num_users: number of users
        :param num_items: number of items
        :param device: choice of device
        """
        super(CDAE, self).__init__()
        self.hidden_dim = model_conf.hidden_dim
        self.act = model_conf.act
        self.corruption_ratio = model_conf.corruption_ratio
        self.num_users = num_users
        self.num_items = num_items
        self.device = device

        self.user_embedding = nn.Embedding(self.num_users, self.hidden_dim)
        self.encoder = nn.Linear(self.num_items, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, self.num_items)

        self.to(self.device)

    def forward(self, user_id, rating_matrix):
        """
        Forward pass
        :param rating_matrix: rating matrix
        """
        # normalize the rating matrix
        user_degree = torch.norm(rating_matrix, 2, 1).view(-1, 1)  # user, 1
        item_degree = torch.norm(rating_matrix, 2, 0).view(1, -1)  # 1, item
        normalize = torch.sqrt(user_degree @ item_degree)
        zero_mask = normalize == 0
        normalize = torch.masked_fill(normalize, zero_mask.bool(), 1e-10)

        normalized_rating_matrix = rating_matrix / normalize

        # corrupt the rating matrix
        normalized_rating_matrix = F.dropout(normalized_rating_matrix, self.corruption_ratio, training=self.training)

        # build the collaborative denoising autoencoder
        enc = self.encoder(normalized_rating_matrix) + self.user_embedding(user_id)
        enc = apply_activation(self.act, enc)
        dec = self.decoder(enc)

        return torch.sigmoid(dec)
