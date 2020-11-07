import numpy as np
import torch
from torch.utils.data import DataLoader

from . import dataset


def ICDARCollectFN(batch):
    data_dict = {}
    to_tensor_keys = set()
    for sample in batch:
        for k, v in sample.items():
            if k not in data_dict:
                data_dict[k] = []
            if isinstance(v, (np.ndarray, torch.Tensor)):
                to_tensor_keys.add(k)
            data_dict[k].append(v)
    for k in to_tensor_keys:
        data_dict[k] = torch.stack(data_dict[k], 0)
      
    return data_dict


def get_dataloader(config: dict, distributed=False):

    if config['loader']['collate_fn'] != 'ICDARCollectFN':
        config['loader']['collate_fn'] = None
    else:
        config['loader']['collate_fn'] = eval(config['loader']['collate_fn'])

    _dataset = getattr(dataset, config['type'])(**config['args'])
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(_dataset)
        config['loader']['shuffle'] = False
        config['loader']['batch_size'] = config['loader']['batch_size'] // torch.cuda.device_count()
    loader = DataLoader(dataset=_dataset, sampler=sampler, **config['loader'])

    return loader, sampler
