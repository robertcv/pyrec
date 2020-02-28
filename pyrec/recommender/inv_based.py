import numpy as np

from pyrec.recommender import BaseRecommender
from pyrec.inventory import Inventory


class MostInInvRecommender(BaseRecommender):
    def __init__(self, inv: Inventory):
        super().__init__()
        self.inv = inv

    def _predict(self, _: int, item_index: int) -> float:
        if self.inv.counts.max() > 1:
            return self.inv.counts[item_index] / self.inv.counts.max() * \
                   self.data.train_data.ratings.max()
        else:
            return self.data.train_data.ratings.max()

    def _predict_user(self, _: int) -> np.ndarray:
        ratings = np.array(self.inv.counts)
        ratings = ratings / ratings.max()
        ratings = ratings * self.data.train_data.ratings.max()
        return ratings


class MostInInvStaticRecommender(BaseRecommender):
    def __init__(self, inv: Inventory):
        super().__init__()
        self.inv = inv
        self.item_pred = None

    def fit(self, data):
        super().fit(data)
        max_r = self.data.train_data.ratings.max()
        self.item_pred = self.inv.counts / self.inv.counts.max() * max_r

    def _predict(self, _: int, item_index: int) -> float:
        return self.item_pred[item_index]

    def _predict_user(self, _: int) -> np.ndarray:
        return self.item_pred


class WeightedRecommender(MostInInvRecommender):
    """
    r = alpha * rec_r + (1 - alpha) * inv_r
    """
    def __init__(self, alpha, inv: Inventory, rec: BaseRecommender,
                 rec_kwargs: dict, verbose=True):
        super().__init__(inv)
        self.rec = rec
        self.alpha = alpha
        self.rec_kwargs = rec_kwargs
        self.verbose = verbose
        self.rec_kwargs["verbose"] = self.verbose

    def fit(self, data):
        super().fit(data)
        if isinstance(self.rec, BaseRecommender):
            # rec is already an instantiated recommender
            return

        self.rec = self.rec(**self.rec_kwargs)
        self.rec.fit(data)

    def __setattr__(self, name, value):
        if name == "data" and "rec" in self.__dict__ and isinstance(self.rec, BaseRecommender):
            self.rec.data = value
        self.__dict__[name] = value

    def _predict(self, user_index: int, item_index: int) -> float:
        r = self.alpha * self.rec._predict(user_index, item_index) + \
            (1 - self.alpha) * super()._predict(user_index, item_index)
        return r

    def _predict_user(self, user_index: int) -> np.ndarray:
        ratings = self.alpha * self.rec._predict_user(user_index) + \
            (1 - self.alpha) * super()._predict_user(user_index)
        return ratings

    def save(self, file_name):
        self.rec.save(file_name)

    def load(self, file_name):
        if not isinstance(self.rec, BaseRecommender):
            self.rec = self.rec(**self.rec_kwargs)
        self.rec.load(file_name)


if __name__ == '__main__':
    from pyrec.data import UIRData
    from pyrec.recommender import MatrixFactorization

    RATINGS_FILE = "../../data/MovieLens/ml-latest-small/ratings.csv"
    uir_data = UIRData.from_csv(RATINGS_FILE)
    inv = Inventory(uir_data)
    mf = MatrixFactorization.load_static("../../models/ml-small-mf")
    mf.data = uir_data

    mr = MostInInvRecommender(inv)
    mr.fit(uir_data)
