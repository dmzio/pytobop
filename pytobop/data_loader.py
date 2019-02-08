import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
import attr
import math

@attr.s
class BaseDataLoaderConfig(object):
    """
    Configuration class for DataLoaders
    """
    batch_size = attr.ib(default=32)
    shuffle = attr.ib(default=True)
    num_workers = attr.ib(default=0)

    validation_split = attr.ib(default=None)
    validation_shuffle = attr.ib(default=False)

    data_dir = attr.ib(default="../data")


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, config: BaseDataLoaderConfig, collate_fn=default_collate):
        self.dataset = dataset
        self.config = config
        self.collate_fn = collate_fn

        self.batch_size = config.batch_size
        self.validation_split = config.validation_split
        self.validation_shuffle = config.validation_shuffle
        self.shuffle = config.shuffle
        self.num_workers = config.num_workers
        
        self.batch_idx = 0
        self._n_samples = len(self.dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
            }
        super(BaseDataLoader, self).__init__(sampler=self.sampler, **self.init_kwargs)

    def __len__(self):
        """
        :return: Total number of batches
        """
        return math.ceil(self._n_samples / self.batch_size)

    def _split_sampler(self, split):
        if not split:
            return None, None

        idx_full = np.arange(self._n_samples)
        if self.validation_shuffle:
            np.random.shuffle(idx_full)

        len_valid = int(self._n_samples * split)

        valid_idx = idx_full[-len_valid:]
        train_idx = idx_full[:-len_valid]
        
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self._n_samples = len(train_idx)

        return train_sampler, valid_sampler
        
    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
    
