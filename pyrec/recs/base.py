from typing import Optional

import numpy as np

from pyrec.data import UIRData


class BaseRecommender:
    def __init__(self, *args, **kwargs):
        self.data = None  # type: Optional[UIRData]

    def fit(self, data: UIRData):
        self.data = data

    def _predict(self, user_index: int, item_index: int) -> float:
        raise NotImplementedError

    def predict(self, user, item):
        return self._predict(self.data.user2index[user],
                             self.data.item2index[item])

    def predict_unknown(self, user, item):
        """Use if user or item or bough are maybe unknown"""
        u = self.data.user2index.get(user, None)
        i = self.data.item2index.get(item, None)

        if u is not None and i is not None:
            return self._predict(u, i)
        elif u is not None:
            return self.data.user_avg[u]
        elif i is not None:
            return self.data.item_avg[i]
        else:
            return self.data.global_avg

    def _predict_user(self, user_index: int) -> np.ndarray:
        raise NotImplementedError

    def predict_user(self, user):
        return self._predict_user(self.data.user2index[user])

    def predict_unknown_user(self, user):
        """Use if user maybe unknown"""
        u = self.data.user2index.get(user, None)

        if u is not None:
            return self._predict_user(u)
        else:
            return self.data.item_avg

    def save(self, file_name):
        raise NotImplemented

    def load(self, file_name):
        raise NotImplemented
