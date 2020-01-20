import numpy as np

from pyrec.recommender import BaseRecommender
from pyrec.inventory import Inventory


class MostInInvRecommender(BaseRecommender):
    def __init__(self, inv: Inventory):
        super().__init__()
        self.inv = inv

    def _predict(self, _: int, item_index: int) -> float:
        return self.inv.counts[item_index] / self.inv.counts.max() * \
               self.data.train_data.ratings.max()

    def _predict_user(self, _: int) -> np.ndarray:
        ratings = np.array(self.inv.counts)
        ratings = ratings / ratings.max()
        ratings = ratings * self.data.train_data.ratings.max()
        return ratings


class WeightedRecommender(MostInInvRecommender):
    """
    r = alpha * rec_r + (1 - alpha) * inv_r
    """
    def __init__(self, inv: Inventory, rec: BaseRecommender, alpha=0.5):
        super().__init__(inv)
        self.rec = rec
        self.alpha = alpha

    def _predict(self, user_index: int, item_index: int) -> float:
        r = self.alpha * self.rec._predict(user_index, item_index) + \
            (1 - self.alpha) * super()._predict(user_index, item_index)
        return r

    def _predict_user(self, user_index: int) -> np.ndarray:
        ratings = self.alpha * self.rec._predict_user(user_index) + \
            (1 - self.alpha) * super()._predict_user(user_index)
        return ratings


if __name__ == '__main__':
    from pyrec.data import UIRData
    from pyrec.recommender import MatrixFactorization

    RATINGS_FILE = "../../data/MovieLens/ml-latest-small/ratings.csv"
    uir_data = UIRData.from_csv(RATINGS_FILE)
    inv = Inventory(uir_data)
    mf = MatrixFactorization.load("../../models/ml-small-mf")
    mf.data = uir_data

    mr = MostInInvRecommender(inv)
    mr.fit(uir_data)
    print(mr.top_n(5))
