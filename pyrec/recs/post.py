import numpy as np

from pyrec.recs.mf import MatrixFactorization


class UnbiasedMatrixFactorization(MatrixFactorization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_max = 0
        self.p_min = 0
        self.t_max = 0
        self.t_min = 0

    def fit(self, data):
        super().fit(data)
        full = np.dot(self.P, self.Q.T)
        self.p_max = np.max(full)
        self.p_min = np.min(full)
        self.t_min = self.data.train_data.ratings.min()
        self.t_max = self.data.train_data.ratings.max()

    def _trans(self, x):
        norm_x = (x - self.p_min) / (self.p_max - self.p_min)
        return norm_x * (self.t_max - self.t_min) + self.t_min

    def _predict(self, user_index, item_index):
        return self._trans(np.sum(self.P[user_index, :] * self.Q[item_index, :]))

    def _predict_user(self, user_index):
        return self._trans(
            np.sum(self.P[np.repeat(user_index, self.data.m), :] *
                   self.Q[np.arange(self.data.m), :], axis=1))


class UnbiasedUsersMatrixFactorization(UnbiasedMatrixFactorization):
    def fit(self, data):
        super().fit(data)
        full = np.dot(self.P, self.Q.T) + self.i_bayes
        self.p_max = np.max(full)
        self.p_min = np.min(full)
        self.t_min = self.data.train_data.ratings.min()
        self.t_max = self.data.train_data.ratings.max()

    def _predict(self, user_index, item_index):
        return self._trans(np.sum(self.P[user_index, :] *
                                  self.Q[item_index, :]) +
                           self.i_bayes[item_index])

    def _predict_user(self, user_index):
        return self._trans(
            np.sum(self.P[np.repeat(user_index, self.data.m), :] *
                   self.Q[np.arange(self.data.m), :], axis=1) + self.i_bayes)


class UserWantMatrixFactorization(MatrixFactorization):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.full = None
        self.t_max = 0
        self.t_min = 0

    def fit(self, data):
        super().fit(data)
        self.full = np.dot(self.P, self.Q.T) + self.i_bayes
        self.t_min = self.data.train_data.ratings.min()
        self.t_max = self.data.train_data.ratings.max()

    def _predict(self, user_index, item_index):
        return self._predict_user(user_index)[item_index]

    def _predict_user(self, user_index):
        user_r = self.full[user_index, :]
        want = np.sum(self.full > user_r, axis=0)
        not_want = 1 - want / want.max()
        return not_want * (self.t_max - self.t_min) + self.t_min


if __name__ == '__main__':
    from pyrec.data import UIRData
    RATINGS_FILE = "../../data/MovieLens/ml-latest-small/ratings.csv"
    uir_data = UIRData.from_csv(RATINGS_FILE)

    mf = UserWantMatrixFactorization(k=20, max_iteration=20,
                                     batch_size=100)
    mf.fit(uir_data)
    mf.predict_user(uir_data.unique_values.users[0])
