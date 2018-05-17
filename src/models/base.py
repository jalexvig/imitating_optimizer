import numpy as np
import torch
from torch import nn

from src import CONFIG


class BaseModel(nn.Module):

    def __init__(self, batch_size=1):

        super().__init__()

        self._setup()

        self.data_gen_train = self.get_data_gen(batch_size, train=True)
        self.criterion = self.get_criterion()

        self.params = list(self.parameters())
        self.optimizer = CONFIG.optimizer_closure_model(self.params)
        self.n_params = sum(np.prod(p.shape) for p in self.params)

    def forward(self, x):

        raise NotImplementedError

    def _setup(self):

        raise NotImplementedError

    def get_data_gen(self, batch_size, train=True):

        raise NotImplementedError

    def get_criterion(self):

        raise NotImplementedError

    def step(self, calc_deltas=True, update_params=True):

        grads = []
        deltas = []
        losses = []

        for t in range(CONFIG.num_steps_model):
            inp, targets = next(self.data_gen_train)

            if isinstance(inp, np.ndarray):
                inp = torch.from_numpy(inp).float()
            if isinstance(targets, np.ndarray):
                targets = torch.from_numpy(targets).float()

            out = self(inp)

            loss = self.criterion(out, targets)

            losses.append(loss)

            self.zero_grad()
            loss.backward()

            grads.append([p.grad.clone() for p in self.params])

            if calc_deltas:
                params0 = [p.clone() for p in self.params]

                self.optimizer.step()
                deltas.append([p1 - p0 for (p1, p0) in zip(self.params, params0)])

                if not update_params:
                    for (p1, p0) in zip(self.params, params0):
                        p1.data = p0.data

        grads = self._proc_deltas(grads)
        deltas = self._proc_deltas(deltas)

        return grads, deltas, losses

    def evaluate(self, batch_size, reset=True):

        if not hasattr(self, 'data_gen_eval') or reset:
            self.data_gen_eval = self.get_data_gen(batch_size, train=False)

        inp, targets = next(self.data_gen_eval)

        out = self(inp)

        loss = self.criterion(out, targets)

        return loss

    def _proc_deltas(self,
                     l: list,
                     transpose: bool=True,
                     trailing_dim: bool=True):
        """
        Transform time index list of potentially disparate data.
        Args:
            l: A list of values for each timestep (4-D).
            transpose: Make features the first axis, time the second axis.
            trailing_dim: Add an axis to the result.

        Returns:

        """

        res = torch.stack([torch.cat([x.reshape(-1) for x in sl]) for sl in l])

        if transpose:
            res = res.transpose(1, 0)

        if trailing_dim:
            res = res[:, :, None]

        return res
