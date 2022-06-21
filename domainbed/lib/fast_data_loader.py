# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torch.utils.data import Sampler, RandomSampler, WeightedRandomSampler, BatchSampler, DataLoader, SubsetRandomSampler

class _InfiniteSampler(Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers):
        super().__init__()

        
        if weights:
            sampler = WeightedRandomSampler(weights,
                replacement=True,
                num_samples=batch_size)
        else:
            sampler = RandomSampler(dataset,
                replacement=True)

        if weights == None:
            weights = torch.ones(len(dataset))

        batch_sampler = BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True)

        self._infinite_iterator = iter(DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError

class InfiniteSubDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers):
        super().__init__()

        indexset = [i for i in range(0, len(dataset), 100)]
        sampler = SubsetRandomSampler(indexset)


        batch_sampler = BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True)

        self._infinite_iterator = iter(DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError

class FastDataLoader:
    """DataLoader wrapper with slightly improved speed by not respawning worker
    processes at every epoch."""
    def __init__(self, dataset, batch_size, num_workers):
        super().__init__()

        batch_sampler = BatchSampler(
            RandomSampler(dataset, replacement=False),
            batch_size=batch_size,
            drop_last=False
        )

        self._infinite_iterator = iter(DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

        self._length = len(batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):  # for _ in range(len(self)/10)
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length

class FastSubDataLoader:
    """DataLoader wrapper with slightly improved speed by not respawning worker
    processes at every epoch."""
    def __init__(self, dataset, batch_size, num_workers):
        super().__init__()

        indexset = [i for i in range(0, len(dataset), 10)]

        batch_sampler = BatchSampler(
            SubsetRandomSampler(indexset),
            batch_size=batch_size,
            drop_last=False
        )

        self._infinite_iterator = iter(DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

        self._length = len(batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):  # for _ in range(len(self)/10)
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length
