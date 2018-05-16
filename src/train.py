import itertools
import os

import torch
from scipy.stats import describe
from torch import nn

from src import CONFIG

STABILITY = 1e-8


class MetaOptimizer(nn.Module):

    def __init__(self):

        super().__init__()

        self.rnn = nn.LSTM(
            input_size=1,
            hidden_size=2,
            batch_first=True
        )

        self.params = list(self.parameters())

        self.optimizer = CONFIG.optimizer_closure_meta(self.params)

        if os.path.isfile(CONFIG.fpath_checkpoint):
            print('Loading previous model')
            self.load_state_dict(torch.load(CONFIG.fpath_checkpoint))

    def forward(self, x, state=None, truth=None):

        # gates bounded by -1, 1
        gates, state = self.rnn(x, state)

        x_contrib = x * gates[:, :, :1]

        if truth is None:

            results = [torch.zeros([x.shape[0], x.shape[-1]])]

            for i in range(x.shape[1]):
                last_contrib = results[-1] * gates[:, i, 1:]
                results.append(x_contrib[:, i, :] + last_contrib)

            res = torch.stack(results[1:], dim=1)
        else:

            last = torch.cat([torch.zeros([truth.shape[0], 1, 1]), truth[:, :-1, :]], dim=1)

            last_contrib = last * gates[:, :, :1]

            res = x_contrib + last_contrib

        return res, state

    def run(self):

        for i in itertools.count():

            if i % CONFIG.freq_save == 0 and i:
                torch.save(self.state_dict(), CONFIG.fpath_checkpoint)
                print('Saved at ', i)

            if CONFIG.num_steps_meta and i == CONFIG.num_steps_meta:
                break

            # This will reinitialize model params
            model = CONFIG.model_class()

            grads, deltas_opt, model_losses = model.step()

            truth = deltas_opt if CONFIG.supply_truth else None
            deltas_pred, _ = self(grads, truth=truth)

            # loss = (deltas_opt - deltas_pred).norm()

            perc_error = (deltas_opt - deltas_pred) / (deltas_opt + STABILITY)
            loss = perc_error.norm()

            if CONFIG.freq_debug and i % CONFIG.freq_debug == 0:
                # Note: model losses will be bad since we are reinitializing model every iteration
                print(i, model_losses[0].item(), model_losses[-1].item(), loss.item())
                print(describe(perc_error.abs().data.numpy(), axis=None))
                print(describe(grads.data.numpy(), axis=None))
                print(describe(deltas_opt.data.numpy(), axis=None))
                print(describe(deltas_pred.data.numpy(), axis=None))

            model.zero_grad()
            self.zero_grad()
            loss.backward()

            self.optimizer.step()
