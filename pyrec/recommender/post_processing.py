import numpy as np

from pyrec.recommender import BaseRecommender
from pyrec.inventory import Inventory


class MostInInvRecommender(BaseRecommender):
    def __init__(self, inv: Inventory):
        super().__init__()
        self.inv = inv

    def _predict(self, user_index: int, item_index: int) -> float:
        rating = self.inv.counts[item_index]
        rating = rating / self.inv.counts.max()
        rating = rating * self.data.train_data.ratings.max()
        return rating

    def _predict_user(self, _: int) -> np.ndarray:
        ratings = self.inv.counts.copy()
        ratings = ratings / ratings.max()
        ratings = ratings * self.data.train_data.ratings.max()
        return ratings


if __name__ == '__main__':
    from pyrec.data import UIRData

    uir_data = UIRData(np.array([1, 1, 2]), np.array([3, 2, 1]),
                       np.array([0, 5, 10]))
    inv = Inventory(np.array([1, 2, 3, 1, 2, 2]))

    mr = MostInInvRecommender(inv)
    mr.fit(uir_data)
    print(mr.top_n(1))
