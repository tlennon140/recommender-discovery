# Import package
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base model class
    """
    def __init__(self):
        super(BaseModel, self).__init__()

    def forward(self, *input):
        pass

    def train_one_epoch(self, *input):
        pass

    def predict(self, eval_users, eval_pos, test_batch_size):
        pass