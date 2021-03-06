import logging
import torch.nn as nn
import numpy as np


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self, config: dict):
        super(BaseModel, self).__init__()
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, *inp):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def summary(self) -> dict:
        """
        Model summary
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info('Trainable parameters: {}'.format(params))
        self.logger.info(self)
        return {'trainable_params_num': params}
