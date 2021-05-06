"""
Harald Steck, Embarrassingly Shallow Autoencoders for Sparse Data. WWW 2019.
https://arxiv.org/pdf/1905.03375
"""
# Import packages
import numpy as np
import torch

# Import utility script
from BaseModel import BaseModel

class ESAE(BaseModel):
    """
    Embarrassingly Shallow Autoencoders model class
    """
    def forward(self, rating_matrix):
        """
        Forward pass
        :param rating_matrix: rating matrix
        """
        G = rating_matrix.transpose(0, 1) @ rating_matrix

        diag = list(range(G.shape[0]))
        G[diag, diag] += self.reg
        P = G.inverse()

        # B = P * (X^T * X − diagMat(γ))
        self.enc_w = P / -torch.diag(P)
        min_dim = min(*self.enc_w.shape)
        self.enc_w[range(min_dim), range(min_dim)] = 0

        # Calculate the output matrix for prediction
        output = rating_matrix @ self.enc_w

        return output