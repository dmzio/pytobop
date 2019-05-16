import os
import math
import json
import logging
import typing
from pathlib import Path
import torch
import torch.optim as optim
from .util import ensure_dir
import attr
from ignite.metrics.metric import Metric
from ignite.engine import Engine, Events
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from .data_loader import BaseDataLoaderConfig
from .model import BaseModel


@attr.s
class BaseOptimizerConfig(object):
    opt_type = attr.ib(default='Adam')
    lr = attr.ib(default=0.001)
    weight_decay = attr.ib(default=0)


@attr.s
class BaseTrainerConfig(object):
    save_freq = attr.ib(default=3)
    epochs = attr.ib(default=10)
    verbosity = attr.ib(default=1)
    save_dir = attr.ib(default='saved')
    monitor = attr.ib(default="val_loss")
    monitor_mode = attr.ib(default="min")


@attr.s
class BaseLRSchedulerConfig(object):
    type = attr.ib(default="ExponentialLR")
    freq = attr.ib(default=1)
    params = attr.ib()

    @params.default
    def default_params(self):
        return {'gamma': 0.87}


@attr.s
class BaseConfig(object):
    """
    Configuration class for whole training workflow setup
    """
    name = attr.ib()
    model = attr.ib()
    arch = attr.ib()
    loss = attr.ib()
    metrics = attr.ib()

    trainer: BaseTrainerConfig = attr.ib(factory=BaseTrainerConfig)
    optimizer: BaseOptimizerConfig = attr.ib(factory=BaseOptimizerConfig)
    cuda = attr.ib(default=True)
    gpu = attr.ib(default=0)

    data_loader: BaseDataLoaderConfig = attr.ib(factory=BaseDataLoaderConfig)
    lr_scheduler: BaseLRSchedulerConfig = attr.ib(factory=BaseLRSchedulerConfig)


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model: BaseModel, loss: typing.Callable, metrics: typing.Dict[str, Metric],
                 config: BaseConfig, resume: typing.Optional[Path] = None, train_logger=None):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.name = config.name
        self.epochs = config.trainer.epochs
        self.save_freq = config.trainer.save_freq
        self.verbosity = config.trainer.verbosity
        self.gpu = None
        if config.gpu is not None:
            if not torch.cuda.is_available():
                self.logger.warning('Warning: There\'s no CUDA support on this machine, '
                                'training is performed on CPU.')
            else:
                self.gpu = torch.device('cuda:' + str(config.gpu))
                self.model = self.model.to(self.gpu)

        self.train_logger = train_logger
        self.optimizer = getattr(optim, config.optimizer.opt_type)(model.parameters(),
                                                                  **attr.asdict(config.optimizer, filter=attr.filters.exclude(attr.fields(BaseOptimizerConfig).opt_type)))
        self.lr_scheduler = getattr(
            optim.lr_scheduler,
            config.lr_scheduler.type, None)
        if self.lr_scheduler:
            self.lr_scheduler = self.lr_scheduler(self.optimizer, **config.lr_scheduler.params)
            self.lr_scheduler_freq = config.lr_scheduler.freq
        self.monitor = config.trainer.monitor
        self.monitor_mode = config.trainer.monitor_mode
        assert self.monitor_mode == 'min' or self.monitor_mode == 'max'
        self.monitor_best = math.inf if self.monitor_mode == 'min' else -math.inf
        self.start_epoch = 1
        self.checkpoint_dir = os.path.join(config.trainer.save_dir, self.name)
        ensure_dir(self.checkpoint_dir)
        json.dump(attr.asdict(config), open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
                  indent=4, sort_keys=False)
        if resume:
            self._resume_checkpoint(resume)

    def _train_update_func(self, engine, batch_values):
        raise NotImplementedError

    def _eval_inference_func(self, engine, batch_values):
        raise NotImplementedError

    def _prepare_evaluator(self):
        if self.gpu:
            self.model.to(self.gpu)

        engine = Engine(self._eval_inference_func)
        @engine.on(Events.STARTED)
        def log_loss_start(engine):
            engine.state.total_loss = 0

        @engine.on(Events.ITERATION_COMPLETED)
        def log_loss(engine):
            engine.state.total_loss += engine.state.output['loss']

        @engine.on(Events.COMPLETED)
        def log_results(engine):
            self.logger.info(f"eval results... {engine.state.metrics}")

        for name, metric in self.metrics.items():
            metric.attach(engine, name)
        return engine

    def run_validate(self, train_engine, evaluate_engine):
        self.model.eval()
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        if self.train_logger is not None:
            self.train_logger.watch(self.model)

        engine = Engine(self._train_update_func)

        @engine.on(Events.EPOCH_STARTED)
        def log_training_loss(engine):
            engine.state.total_loss = 0

        @engine.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine):
            engine.state.total_loss += engine.state.output['loss']

        for name, metric in self.metrics.items():
            metric.attach(engine, name)

        pbar = ProgressBar()
        pbar.attach(engine)

        if self.valid:  # TODO proper implementation: currently handled only in subclass
            evaluator = self._prepare_evaluator()
            engine.add_event_handler(Events.EPOCH_COMPLETED, self.run_validate, evaluator)

        @engine.on(Events.EPOCH_COMPLETED)
        def mk_checkpoints(engine):  # TODO use checkpointing/scheduling from ignnite
            log = {
                'epoch': engine.state.epoch,
                'loss': engine.state.total_loss / len(engine.state.dataloader),
                'metrics': engine.state.metrics
            }
            if hasattr(engine.state, 'validation_result'):
                log['val_loss'] = engine.state.validation_result.total_loss / len(engine.state.validation_result.dataloader)

            self._prepare_checkpoint(log=log)
            self._reschedule_lr(epoch=engine.state.epoch)
            if self.train_logger is not None:
                self.train_logger.add_entry(log)
                if self.verbosity >= 1:
                    for key, value in log.items():
                        self.logger.info('    {:15s}: {}'.format(str(key), value))

        engine.run(self.data_loader, max_epochs=self.epochs)  # TODO return resume logic of range(self.start_epoch, self.epochs + 1):

    def _reschedule_lr(self, epoch):
        if self.lr_scheduler and epoch % self.lr_scheduler_freq == 0:
            self.lr_scheduler.step(epoch)
            lr = self.lr_scheduler.get_lr()[0]
            self.logger.info('New Learning Rate: {:.6f}'.format(lr))

    def _prepare_checkpoint(self, log: dict):
        epoch = log['epoch']
        if (self.monitor_mode == 'min' and log[self.monitor] < self.monitor_best) \
                or (self.monitor_mode == 'max' and log[self.monitor] > self.monitor_best):
            self.monitor_best = log[self.monitor]
            self._save_checkpoint(epoch, log, save_best=True)
        if epoch % self.save_freq == 0:
            self._save_checkpoint(epoch, log)

    def _train_epoch(self, epoch) -> dict:
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _save_checkpoint(self, epoch, log, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth.tar'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'logger': self.train_logger,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.monitor_best,
            'config': self.config
        }
        if 'val_loss' in log:
            filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{:03d}-val_loss-{:.4f}.pth.tar'
                                    .format(epoch, log['val_loss']))
        else:
            filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{:03d}-loss-{:.4f}.pth.tar'
                                    .format(epoch, log['loss']))
        torch.save(state, filename)
        if save_best:
            os.rename(filename, os.path.join(self.checkpoint_dir, 'model_best.pth.tar'))
            self.logger.info("Saving current best: {} ...".format('model_best.pth.tar'))
        else:
            self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.gpu:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(self.gpu)
        self.train_logger = checkpoint['logger']
        raise NotImplementedError
        self.config = checkpoint['config']
        self.logger.info("Checkpoint '{}' (epoch {}) loaded".format(resume_path, self.start_epoch))
