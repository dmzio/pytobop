"""
Helper functions to configure and run experiments
"""
import logging
import argparse
import torch
from pathlib import Path
import shutil
import json
from cattr import structure
from .trainer import BaseConfig


def prepare_config_from_cli(config_class: type(BaseConfig), config_defaults: dict = None,
                            cli_description: str = "PyTorch experiment"):
    """
    Helper to use in Experiment CLI scripts

    Parses CLI arguments and creates Config instance for the experiment
    :param config_class: class of the Config for experiment
    :param config_defaults: if Config has required arguments in `__init__` - provide them here as a dict
    :param cli_description: Description string shown in the help message of CLI script
    :return: instance of Config and 'resume' bool flag
    """
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description=cli_description)
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite previous results (for draft experiments)')

    args = parser.parse_args()

    if args.resume is not None:
        if args.config is not None:
            logger.warning('Warning: --config overridden by --resume')
        config = torch.load(args.resume)['config']
    elif args.config:
        config_path = Path(args.config)
        config = read_config_from_json(config_class, config_path)
    else:
        config = config_class(**config_defaults) if config_defaults is not None else config_class()

    experiment_path = Path(config.trainer.save_dir) / Path(config.name)
    if not args.overwrite:
        assert not experiment_path.exists(), "Path {} already exists!".format(experiment_path)
    elif experiment_path.exists():
        shutil.rmtree(experiment_path)

    return config, args.resume


def read_config_from_json(config_class: type(BaseConfig), config_path: Path):
    """
    Reads provided JSON file and returns Config of given class
    :param config_class: class of the Config for experiment
    :param config_path: Path to JSON file of config
    :return: instance of Config
    """
    config_dict = json.loads(config_path.read_text())
    config = structure(config_dict, config_class)
    return config


