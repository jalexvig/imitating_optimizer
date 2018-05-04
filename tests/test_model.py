import torch

from src.models.base import BaseModel


class TestBaseModel(object):

    def test_proc_deltas(self):

        # 2 timesteps, params of shape (3, 5) and (7, 1)
        data = torch.arange(44).reshape(2, 22)
        l = [(x[:15].reshape(3, 5), x[15:].reshape(7, 1)) for x in data.clone()]

        res = BaseModel._proc_deltas(None, l)

        answer = data.transpose(0, 1)[:, :, None]

        print(res.shape, answer.shape)

        assert (res == answer).all()
