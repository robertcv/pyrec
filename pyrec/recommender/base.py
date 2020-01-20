from typing import Optional

import numpy as np

from pyrec.data import UIRData


class BaseRecommender:

    def __init__(self):
        self.data = None  # type: Optional[UIRData]

    def fit(self, data: UIRData):
        self.data = data

    def _predict(self, user_index: int, item_index: int) -> float:
        raise NotImplementedError

    def predict(self, user, item):
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

    def _top_n_indexes(self, pred, n):
        # TODO: if multiple values are the same we have to randomly
        # select which to put in topN
        return np.argsort(pred)[-n:][::-1]

    def top_n(self, user, n=5):
        u = self.data.user2index.get(user, None)

        if u is not None:
            pred = self._predict_user(u)
        else:
            pred = self.data.item_avg

        top_n = self._top_n_indexes(pred, n)
        return self.data.unique_values.items[top_n], pred[top_n]
