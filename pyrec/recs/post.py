import numpy as np

from pyrec.recs.base import BaseRecommender
from pyrec.recs.mf import MatrixFactorization


class BasePost(BaseRecommender):
    def __init__(self, rec, rec_kwargs, verbose=True, **kwargs):
        self.verbose = verbose

        self.rec = rec  # type: BaseRecommender
        self.rec_kwargs = rec_kwargs
        self.rec_kwargs["verbose"] = self.verbose

        self.rec_fitted = True

        if not isinstance(self.rec, MatrixFactorization):
            self.rec_fitted = False
            self.rec = self.rec(**self.rec_kwargs)

        super().__init__()

    def fit(self, data):
        super().fit(data)
        if not self.rec_fitted:
            self.rec.fit(data)

    def __setattr__(self, name, value):
        if name == "data":
            self.rec.data = value
        self.__dict__[name] = value

    def save(self, file_name):
        self.rec.save(file_name + "_1")

    def load(self, file_name):
        self.rec.load(file_name + "_1")


class UserWantMatrixFactorization(BasePost):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full = None

    def _pre_compute(self):
        self.full = np.dot(self.rec.P, self.rec.Q.T)

    def _predict(self, user_index, item_index):
        if self.full is None:
            self._pre_compute()
        user_r = self.full[user_index, item_index]
        want = np.sum(self.full[:, item_index] > user_r)
        return want

    def _predict_user(self, user_index):
        if self.full is None:
            self._pre_compute()
        user_r = self.full[user_index, :]
        want = np.sum(self.full > user_r, axis=0)
        return want


class UserNotWantMatrixFactorization(UserWantMatrixFactorization):
    def _predict(self, user_index, item_index):
        return self.data.n - super()._predict(user_index, item_index)

    def _predict_user(self, user_index):
        return self.data.n - super()._predict_user(user_index)


if __name__ == '__main__':
    from pyrec.data import UIRData
    RATINGS_FILE = "../../data/MovieLens/ml-latest-small/ratings.csv"
    uir_data = UIRData.from_csv(RATINGS_FILE)

    mf = UserWantMatrixFactorization(k=20, max_iteration=200,
                                     batch_size=100)
    mf.fit(uir_data)
    mf.predict(uir_data.unique_values.users[0],
               uir_data.unique_values.items[0])
    mf.predict_user(uir_data.unique_values.users[0])
