import numpy as np
from torch import nn

from src.models.base import BaseModel


class BinaryClassifierModel(BaseModel):

    def forward(self, x):

        x = self.fc1(x)
        x = self.fc2(x)

        x = x[:, 0]

        return x

    def _setup(self):

        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def get_data_gen(self, batch_size):

        while 1:

            data = np.random.rand(batch_size, 2) * 5
            labels = (data.sum(axis=1) > 5).astype(int)

            yield data, labels

    def get_criterion(self):

        return nn.BCEWithLogitsLoss()
