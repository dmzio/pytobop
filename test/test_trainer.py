import pytest
from unittest.mock import MagicMock
from torch.nn.functional import mse_loss
from pytobop import trainer
from .test_model import model


def test_training(model):
    config = trainer.BaseConfig(
        name='',
        model=None,
        arch=None,
        loss=None,
        metrics=None
    )
    config.trainer.save_dir = '/tmp'
    t = trainer.BaseTrainer(model=model,
                            loss=mse_loss,
                            metrics=[],
                            config=config)
    t._train_epoch = MagicMock(return_value={'val_loss': 1})
    t.train()
    t._train_epoch.assert_called()
    assert False