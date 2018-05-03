import itertools

import numpy as np

from src import CONFIG
from src.train import MetaOptimizer


def test():

    meta_learner = MetaOptimizer()

    model = CONFIG.model_class()

    state = None

    for i in itertools.count():
        grads, deltas_opt, losses = model.step(update_params=False)

        deltas_pred, state = meta_learner(grads, state)

        l = (deltas_opt - deltas_pred).norm()

        perc_diff = (deltas_opt - deltas_pred) / (deltas_opt + 1e-8)

        deltas_pred = deltas_pred[:, 0, 0]

        if i % 100 == 0:
            print(i, l.item(), losses[0].item(), perc_diff.view(-1).abs().max().item())
            # from scipy.stats import describe
            # print(describe(diff.data.numpy(), axis=None))

        j = 0
        for param in model.params:
            size = np.prod(param.shape)
            delta = deltas_pred[j: j + size].reshape(param.shape)
            param.data.add_(delta)
