import numpy as np

from pyrec.recommender import BaseRecommender
from pyrec.inventory import Inventory


class MostInInvRecommender(BaseRecommender):
    def __init__(self, inv: Inventory, rec: BaseRecommender):
        super().__init__()
        self.inv = inv
        self.rec = rec

    def _predict(self, user_index: int, item_index: int) -> float:
        return self.rec._predict(user_index, item_index)

    def top_n(self, user, n=5):
        # sort by most in inventory
        top_n = np.argsort(self.inv.counts)[-n:][::-1]

        u = self.data.user2index.get(user, None)
        if u is not None:
            pred = self.rec._predict_user(u)
        else:
            pred = self.data.item_avg

        return self.data.unique_values.items[top_n], pred[top_n]


if __name__ == '__main__':
    from pyrec.data import UIRData
    from pyrec.recommender import MatrixFactorization

    RATINGS_FILE = "../../data/MovieLens/ml-latest-small/ratings.csv"
    uir_data = UIRData.from_csv(RATINGS_FILE)
    inv = Inventory(uir_data)
    mf = MatrixFactorization.load("../../models/ml-small-mf")
    mf.data = uir_data

    mr = MostInInvRecommender(inv, mf)
    mr.fit(uir_data)
    print(mr.top_n(5))
