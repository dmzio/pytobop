import pytest
import torch.nn as nn
from pytobop.model import BaseModel


class SimpleModel(BaseModel):
    def __init__(self, config: dict):
        super(SimpleModel, self).__init__(config)
        self.config = config
        self.out = nn.Linear(1, 1)

    def forward(self, x):
        return self.out(x)


@pytest.fixture
def model():
    return SimpleModel({})


def test_model_summary(model):
    summary = model.summary()
    assert summary['trainable_params_num'] == 2
