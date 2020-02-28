import numpy as np

from pyrec.recommender import BaseRecommender
from pyrec.inventory import Inventory


class MostInInvRecommender(BaseRecommender):
    """
    Always assign the highest rating to item that we have the
    most of in inventory. Changes when inventory changes.
    """
    def __init__(self, inv: Inventory, **kwargs):
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

    def save(self, file_name):
        """
        Inventory cannot by saved since this object must be the
        same everywhere it's manipulated.
        """
        pass

    def load(self, file_name):
        """
        Inventory cannot by loaded since this object must be the
        same everywhere it's manipulated.
        """
        pass


class MostInInvStaticRecommender(BaseRecommender):
    """
    Static version of MostInInvRecommender. Ratings are computed once and
    are not updated when inventory changes.
    """
    def __init__(self, inv: Inventory, **kwargs):
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

    def save(self, file_name):
        if not file_name.endswith(".npz"):
            file_name += ".npz"
        np.savez(file_name, item_pred=self.item_pred)

    def load(self, file_name):
        if not file_name.endswith(".npz"):
            file_name += ".npz"
        data = np.load(file_name)
        self.item_pred = data["item_pred"]


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
