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


class UnbiasedMatrixFactorization(BasePost):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.rec, MatrixFactorization)
        self.p_max = 0
        self.p_min = 0
        self.t_max = 0
        self.t_min = 0

    def _pre_compute(self):
        full = np.dot(self.rec.P, self.rec.Q.T)
        self.p_max = full.max()
        self.p_min = full.min()
        self.t_min = self.data.train_data.ratings.min()
        self.t_max = self.data.train_data.ratings.max()

    def _trans(self, x):
        if self.p_max == 0:
            self._pre_compute()
        norm_x = (x - self.p_min) / (self.p_max - self.p_min)
        return norm_x * (self.t_max - self.t_min) + self.t_min

    def _predict(self, user_index, item_index):
        return self._trans(
            np.sum(self.rec.P[user_index, :] * self.rec.Q[item_index, :]))

    def _predict_user(self, user_index):
        return self._trans(
            np.sum(self.rec.P[np.repeat(user_index, self.data.m), :] *
                   self.rec.Q[np.arange(self.data.m), :], axis=1))


class UnbiasedUsersMatrixFactorization(UnbiasedMatrixFactorization):
    def _pre_compute(self):
        full = np.dot(self.rec.P, self.rec.Q.T) + self.rec.i_bayes
        self.p_max = full.max()
        self.p_min = full.min()
        self.t_min = self.data.train_data.ratings.min()
        self.t_max = self.data.train_data.ratings.max()

    def _predict(self, user_index, item_index):
        return self._trans(np.sum(self.rec.P[user_index, :] *
                                  self.rec.Q[item_index, :]) +
                           self.rec.i_bayes[item_index])

    def _predict_user(self, user_index):
        return self._trans(
            np.sum(self.rec.P[np.repeat(user_index, self.data.m), :] *
                   self.rec.Q[np.arange(self.data.m), :], axis=1) +
            self.rec.i_bayes)


class UserNotWantMatrixFactorization(BasePost):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.rec, MatrixFactorization)
        self.full = None
        self.t_max = 0
        self.t_min = 0

    def _pre_compute(self):
        self.full = np.dot(self.rec.P, self.rec.Q.T)
        self.t_min = self.data.train_data.ratings.min()
        self.t_max = self.data.train_data.ratings.max()

    def _trans(self, x):
        return x * (self.t_max - self.t_min) + self.t_min

    def _predict(self, user_index, item_index):
        if self.full is None:
            self._pre_compute()
        user_r = self.full[user_index, item_index]
        want = np.sum(self.full[:, item_index] > user_r)
        not_want = 1 - want / self.data.n
        return self._trans(not_want)

    def _predict_user(self, user_index):
        if self.full is None:
            self._pre_compute()
        user_r = self.full[user_index, :]
        want = np.sum(self.full > user_r, axis=0)
        not_want = 1 - want / self.data.n
        return self._trans(not_want)


class UserWantMatrixFactorization(UserNotWantMatrixFactorization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full = None
        self.t_max = 0
        self.t_min = 0

    def _predict(self, user_index, item_index):
        if self.full is None:
            self._pre_compute()
        user_r = self.full[user_index, item_index]
        want = np.sum(self.full[:, item_index] > user_r)
        return self._trans(want / self.data.n)

    def _predict_user(self, user_index):
        if self.full is None:
            self._pre_compute()
        user_r = self.full[user_index, :]
        want = np.sum(self.full > user_r, axis=0)
        return self._trans(want / self.data.n)


if __name__ == '__main__':
    from pyrec.data import UIRData
    RATINGS_FILE = "../../data/MovieLens/ml-latest-small/ratings.csv"
    uir_data = UIRData.from_csv(RATINGS_FILE)

    mf = UnbiasedMatrixFactorization(k=20, max_iteration=200,
                                     batch_size=100)
    mf.fit(uir_data)
    mf.predict(uir_data.unique_values.users[0],
               uir_data.unique_values.items[0])
    mf.predict_user(uir_data.unique_values.users[0])
