"""Deterministic distributed window sampling, including exact resume offsets."""
from collections import Counter
import torch
from torch.utils.data import Sampler


class WindowSampler(Sampler):
    def __init__(self, dataset, *, world=1, rank=0, seed=0, epoch=0,
                 mode='window_uniform', offset=0):
        self.dataset, self.world, self.rank = dataset, world, rank
        self.seed, self.epoch, self.mode, self.offset = seed, epoch, mode, offset
        if mode not in ['window_uniform', 'task_uniform', 'level_balanced']:
            raise ValueError(mode)
        self.size = len(dataset)//world

    def __len__(self):
        return max(0, self.size-self.offset)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        n = self.size*self.world
        if self.mode == 'window_uniform':
            indices = torch.randperm(len(self.dataset), generator=generator)[:n]
        else:
            tasks = [self.dataset.episodes[e]['task'] for e, _ in self.dataset.index]
            counts = Counter(tasks)
            if self.mode == 'task_uniform':
                mass = {t:1/len(counts) for t in counts}
            else:
                levels = Counter(t.split('-')[1] for t in counts)
                level_mass = {'L0':.25, 'L1':.375, 'L2':.375}
                mass = {t:level_mass[t.split('-')[1]]/levels[t.split('-')[1]] for t in counts}
            weights = torch.tensor([mass[t]/counts[t] for t in tasks], dtype=torch.double)
            indices = torch.multinomial(weights, n, replacement=True, generator=generator)
        yield from indices[self.rank:n:self.world][self.offset:].tolist()
