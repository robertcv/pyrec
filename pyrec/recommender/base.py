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

    @staticmethod
    def _top_n_indexes(pred, n):
        arg_sort = np.argsort(pred)[::-1]
        if pred[arg_sort[n - 1]] != pred[arg_sort[n]]:
            return arg_sort[:n]

        # for multiple values chose at random
        top_arg_sort = arg_sort[:n]
        same_pred = pred[arg_sort[n]]
        same_size = np.sum(pred[top_arg_sort] == same_pred)
        same_indexes = np.where(pred == same_pred)[0]
        same_random = np.random.choice(same_indexes, same_size, replace=False)
        top_arg_sort[-same_size:] = same_random
        return top_arg_sort

    def top_n(self, user, n=5):
        u = self.data.user2index.get(user, None)

        if u is not None:
            pred = self._predict_user(u)
        else:
            pred = self.data.item_avg

        top_n = self._top_n_indexes(pred, n)
        return self.data.unique_values.items[top_n], pred[top_n]


if __name__ == '__main__':
    pred = np.array([5, 5, 4, 3, 3, 3, 3, 3, 2, 2, 1])
    np.random.shuffle(pred)
    BaseRecommender._top_n_indexes(pred, 5)
