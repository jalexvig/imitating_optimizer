import itertools

import torch
from scipy.stats import describe

import numpy as np

from src import CONFIG
from src.train import MetaOptimizer

WEIGHT_DECAY = 1e-3


def test():

    update_params = False

    meta_learner = MetaOptimizer()

    model = CONFIG.model_class()

    state = None

    for i in itertools.count():

        grads, deltas_opt, losses = model.step(update_params=update_params)

        deltas_pred, state = meta_learner(grads, state)

        params = torch.cat([p.reshape(-1) for p in model.params])

        l = (deltas_opt - deltas_pred).norm()

        perc_diff = (deltas_opt - deltas_pred) / (deltas_opt + 1e-8)

        if torch.isnan(l).any():
            params_stats = describe(params.abs().data.numpy(), axis=None)
            print(i, params_stats)
            import ipdb; ipdb.set_trace()

        if i % 100 == 0:
            stats = describe(perc_diff.abs().data.numpy(), axis=None)
            params_stats = describe(params.abs().data.numpy(), axis=None)
            print(i,
                  l.item(),
                  losses[0].item(),
                  (stats.minmax, stats.mean, stats.variance),
                  (params_stats.minmax, params_stats.mean, params_stats.variance),
                  )

        if update_params:
            continue

        j = 0
        for param in model.params:
            size = np.prod(param.shape)
            # deltas_pred = grads.view(-1) * -0.01
            delta = deltas_pred[j: j + size].reshape(param.shape)
            delta -= WEIGHT_DECAY * params[j: j + size].reshape(param.shape)
            param.data.add_(delta)
            j += size
