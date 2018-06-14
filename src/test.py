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

    results = []

    for i in itertools.count():

        results.append(model.evaluate(64).item())

        if i == 4000:
            break

        grads, deltas_opt, model_losses = model.step(update_params=update_params)

        deltas_pred, state = meta_learner(grads, state)

        params = torch.cat([p.reshape(-1) for p in model.params])

        l = (deltas_opt - deltas_pred).norm()

        if torch.isnan(l).any():
            params_stats = describe(params.abs().data.numpy(), axis=None)
            print(i, params_stats)
            import ipdb; ipdb.set_trace()

        if i % 100 == 0:
            perc_diff = (deltas_opt - deltas_pred) / (deltas_opt + 1e-8)

            stats = describe(perc_diff.abs().data.numpy(), axis=None)
            params_stats = describe(params.abs().data.numpy(), axis=None)
            print(i,
                  l.item(),
                  model_losses[0].item(),
                  (stats.minmax, stats.mean, stats.variance),
                  (params_stats.minmax, params_stats.mean, params_stats.variance),
                  )

        if update_params:
            continue

        j = 0
        for param in model.params:
            size = np.prod(param.shape)
            # delta = -0.01 * grads[j: j + size].reshape(param.shape)
            delta = deltas_pred[j: j + size].reshape(param.shape)
            delta -= WEIGHT_DECAY * params[j: j + size].reshape(param.shape)
            param.data.add_(delta)
            j += size

    return results


def graph(results, title=''):

    import seaborn as sns
    from matplotlib import pyplot as plt
    from scipy.signal import savgol_filter

    sns.tsplot(savgol_filter(results, 31, 2), color=sns.xkcd_rgb['pale red'])
    plt.scatter(np.arange(len(results)), results, s=2)

    plt.xlabel('Iteration')
    plt.ylabel('Model Loss')

    if title:
        plt.title(title)

    plt.show()


if __name__ == '__main__':

    from main import proc_flags
    CONFIG.test = '/Users/alex/ml/lstm_learn_optimizer/saved/multivargauss_binary_adam_sgd_1/config.txt'
    proc_flags()
    CONFIG.num_steps_model = 1

    results = test()
    graph(results, 'multivariate gaussian binary classifier')
