import argparse
import json
import os
import shutil
import functools

import torch
from torch.optim import SGD, Adadelta, Adam

from src import CONFIG, models, train


def parse_flags():

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='binary', choices=['binary', 'mnist'], help='Name of model.')
    parser.add_argument('--model_dir', default='./saved', help='Directory to save run information in.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--optimizer_model', choices=['sgd', 'adadelta', 'adam'], default='sgd',
                        help='Number of steps to run model')
    parser.add_argument('--optimizer_meta', choices=['sgd', 'adadelta', 'adam'], default='adam',
                        help='Number of steps to run model')
    parser.add_argument('--optimizer_kwargs_model',
                        help='Comma separated kwargs for model optimizer. Keys and values are separated by a colon. E.g. lr:0.01,alpha:0.2')
    parser.add_argument('--optimizer_kwargs_meta',
                        help='Comma separated kwargs for meta-model optimizer. Keys and values are separated by a colon. E.g. lr:0.01,alpha:0.2')
    parser.add_argument('--num_steps_model', default=50, type=int, help='Number of steps to run model.')
    parser.add_argument('--num_steps_meta', default=0, type=int, help='Number of steps to run meta-model.')
    parser.add_argument('--reset', action='store_true',
                        help='If set, delete the existing model directory and start training from scratch.')
    parser.add_argument('--supply_truth', action='store_true', help='Supply optimizer deltas to meta-optimizer.')
    parser.add_argument('--run_name', default='default', help='Name of run.')
    parser.add_argument('--test', help='Dir path for trained model/config file. THIS OVERRIDES OTHER OPTIONS.')
    parser.add_argument('--freq_save', default=1000, type=int, help='Number of meta steps before saving learner.')
    parser.add_argument('--freq_debug', type=int, help='Number of meta steps before outputtig diagnostic info.')

    parser.parse_args(namespace=CONFIG)


def proc_flags():

    if CONFIG.test:

        if not os.path.isfile(CONFIG.test):
            raise ValueError(CONFIG.test)

        with open(CONFIG.test) as f:
            config = json.load(f)

        for k, v in config.items():
            if k == 'test':
                continue
            setattr(CONFIG, k, v)
    else:
        run_name_components = [CONFIG.run_name,
                               CONFIG.model,
                               str(CONFIG.optimizer_meta),
                               str(CONFIG.optimizer_model),
                               str(CONFIG.seed)]
        CONFIG.dpath_model = os.path.join(CONFIG.model_dir, '_'.join(run_name_components))

        if CONFIG.reset:
            shutil.rmtree(CONFIG.dpath_model, ignore_errors=True)

        CONFIG.fpath_checkpoint = os.path.join(CONFIG.dpath_model, 'checkpoint')
        CONFIG.fpath_eval_data = os.path.join(CONFIG.dpath_model, 'eval_data.pkl')

        if not os.path.exists(CONFIG.dpath_model):
            os.makedirs(CONFIG.dpath_model)

        with open(os.path.join(CONFIG.dpath_model, 'config.txt'), 'w') as f:
            json.dump(vars(CONFIG), f, indent=4)

    torch.manual_seed(CONFIG.seed)

    CONFIG.optimizer_closure_model = get_optimizer_closure(CONFIG.optimizer_model, CONFIG.optimizer_kwargs_model)
    CONFIG.optimizer_closure_meta = get_optimizer_closure(CONFIG.optimizer_meta, CONFIG.optimizer_kwargs_meta)

    if CONFIG.model == 'binary':
        CONFIG.model_class = models.BinaryClassifier
    elif CONFIG.model == 'mnist':
        CONFIG.model_class = models.MNIST


def get_optimizer_closure(name, opts):

    kwargs = dict(kv.split(':') for kv in opts.split(',')) if opts else {}

    for k, v in kwargs.items():
        try:
            kwargs[k] = float(v)
        except ValueError:
            pass

    c = {
        'sgd': SGD,
        'adadelta': Adadelta,
        'adam': Adam,
    }.get(name.lower())

    if not c:
        raise ValueError(name)

    optimizer_closure = functools.partial(c, **kwargs)

    return optimizer_closure


if __name__ == '__main__':

    parse_flags()
    proc_flags()

    if CONFIG.test:
        from src.test import test, graph
        CONFIG.num_steps_model = 1
        results = test()
        graph(results)
    else:
        meta_learner = train.MetaOptimizer()
        meta_learner.run()
