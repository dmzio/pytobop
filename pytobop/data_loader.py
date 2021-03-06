import attr
import math
import numpy as np
import psutil
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class CachedDataset(Dataset):
    def __init__(self, reserved_ram: int = 3):
        """
        Base class which has RAM caching of datapoints
        :param reserved_ram: number of GB to keep free in memory during cache population. If `None` - disables the cache
        """
        self.reserved_ram = reserved_ram
        self._ram_cache = {}

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        if self.reserved_ram is not None and (idx in self._ram_cache):
            return self._ram_cache[idx]

        rv = self._get_item_uncached(idx)

        if self.reserved_ram is not None:
            free_ram_avail = (psutil.virtual_memory().free + psutil.virtual_memory().cached) // (1024 ** 3)
            if free_ram_avail > self.reserved_ram:
                self._ram_cache[idx] = rv
        return rv

    def _get_item_uncached(self, idx):
        raise NotImplementedError


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
    def __init__(self, dataset: Dataset, config: BaseDataLoaderConfig, collate_fn=default_collate):
        """
        Base class for all data loaders
        :param dataset: Dataset to iterate over
        :param config:
        :param collate_fn:
        """
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
