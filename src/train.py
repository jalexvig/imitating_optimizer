import itertools

import os
import torch
from torch import nn

from src import CONFIG


class MetaOptimizer(nn.Module):

    def __init__(self):

        super().__init__()

        self.rnn = nn.LSTM(
            input_size=1,
            hidden_size=1,
            batch_first=True
        )

        self.params = list(self.parameters())

        self.optimizer = CONFIG.optimizer_closure_meta(self.params)

        if os.path.isfile(CONFIG.fpath_checkpoint):
            print('Loading previous model')
            self.load_state_dict(torch.load(CONFIG.fpath_checkpoint))

    def forward(self, x, state=None):

        o, state = self.rnn(x, state)

        # Since o bounded by -1, 1
        res = x * o

        return res, state

    def run(self):

        for i in itertools.count():

            if i % CONFIG.save_freq == 0 and i:
                torch.save(self.state_dict(), CONFIG.fpath_checkpoint)
                print('Saved at ', i)

            if i == CONFIG.num_steps_meta and CONFIG.num_steps_meta:
                break

            model = CONFIG.model_class()

            grads, deltas_opt, losses = model.step()

            deltas_pred, _ = self(grads)

            # loss = (deltas_opt - deltas_pred).norm()

            STABILITY = 1e-8
            perc_error = (deltas_opt - deltas_pred) / (deltas_opt + STABILITY)
            loss = perc_error.norm()

            if i % 20 == 0:
                from scipy.stats import describe
                print(describe(perc_error.abs().data.numpy(), axis=None))
                print(describe(grads.data.numpy(), axis=None))
                print(describe(deltas_opt.data.numpy(), axis=None))
                print(describe(deltas_pred.data.numpy(), axis=None))
                print(i, loss.item())

            model.zero_grad()
            self.zero_grad()
            loss.backward()

            self.optimizer.step()
