import time
from typing import Optional

import numpy as np

from pyrec.recs.base import BaseRecommender
from pyrec.data import UIRData


class MatrixFactorization(BaseRecommender):

    def __init__(self, k=20, max_iteration=50, batch_size=10000,
                 alpha=0.001, mi=0.0001, verbose=True, seed=None, **kwargs):
        super().__init__()

        self.k = k
        self.max_iteration = max_iteration
        self.batch_size = batch_size
        self.alpha = alpha
        self.mi = mi
        self.verbose = verbose
        self.seed = seed

        self.P = None  # type: Optional[np.ndarray]
        self.Q = None  # type: Optional[np.ndarray]
        self.u_bayes = None  # type: Optional[np.ndarray]
        self.i_bayes = None  # type: Optional[np.ndarray]

    def _print_verbose(self, out):
        if self.verbose:
            print(out)

    def fit(self, data: UIRData):
        super().fit(data)
        np.random.seed(self.seed)

        train_u = data.train_data.users
        train_i = data.train_data.items
        train_r = data.train_data.ratings

        validation_u = data.validation_data.users
        validation_i = data.validation_data.items
        validation_r = data.validation_data.ratings

        train_len = len(train_u)
        train_select = np.arange(train_len)

        self.P = np.random.normal(size=(data.n, self.k), scale=1/self.k)
        self.Q = np.random.normal(size=(data.m, self.k), scale=1/self.k)

        self.u_bayes = np.random.normal(size=(data.n,), scale=.1/self.k)
        self.i_bayes = np.random.normal(size=(data.m,), scale=.1/self.k)

        n_batches = np.clip(int(train_len / self.batch_size), 1, train_len)
        self._print_verbose(f"Number of Batches: {n_batches}")

        last_e = np.mean(np.abs(validation_r -
                                self._pred_vec(validation_u, validation_i)))

        for iteration in range(self.max_iteration):
            start_t = time.time()
            np.random.shuffle(train_select)

            train_e = 0
            for batch_i in np.array_split(train_select, n_batches):
                # update P and Q
                u, i, r = train_u[batch_i], train_i[batch_i], train_r[batch_i]
                e = r - self._pred_vec(u, i)
                self.P[u, :] += self.alpha * (e[:, np.newaxis] * self.Q[i, :] - self.mi * self.P[u, :])
                self.Q[i, :] += self.alpha * (e[:, np.newaxis] * self.P[u, :] - self.mi * self.Q[i, :])
                self.u_bayes[u] += self.alpha * (e - self.mi * self.u_bayes[u])
                self.i_bayes[i] += self.alpha * (e - self.mi * self.i_bayes[i])
                train_e += np.sum(np.abs(e))
            train_e /= train_len

            # calculate error on validation data
            validation_e = np.mean(np.abs(validation_r -
                                          self._pred_vec(validation_u, validation_i)))
            iter_t = time.time() - start_t
            self._print_verbose(f"iter t: {iter_t:.5f}, train e: {train_e:.5f}, validation e: {validation_e:.5f}")
            if last_e < validation_e:
                self._print_verbose(f"Ending after iteration {iteration}")
                break
            else:
                last_e = validation_e

    def _pred_vec(self, u, i):
        return np.sum(self.P[u, :] * self.Q[i, :], axis=1) \
               + self.u_bayes[u] + self.i_bayes[i]

    def _predict(self, user_index: int, item_index: int) -> float:
        return self._pred_vec([user_index], [item_index])[0]

    def _predict_user(self, user_index):
        return self._pred_vec(np.repeat(user_index, self.data.m),
                              np.arange(self.data.m))

    def save(self, file_name):
        if not file_name.endswith(".npz"):
            file_name += ".npz"
        np.savez(file_name, P=self.P, Q=self.Q,
                 u_bayes=self.u_bayes, i_bayes=self.i_bayes)

    def load(self, file_name):
        if not file_name.endswith(".npz"):
            file_name += ".npz"
        data = np.load(file_name)
        self.P = data["P"]
        self.Q = data["Q"]
        self.u_bayes = data["u_bayes"]
        self.i_bayes = data["i_bayes"]

    @staticmethod
    def load_static(file_name) -> 'MatrixFactorization':
        if not file_name.endswith(".npz"):
            file_name += ".npz"
        data = np.load(file_name)
        mf = MatrixFactorization()
        mf.P = data["P"]
        mf.Q = data["Q"]
        mf.u_bayes = data["u_bayes"]
        mf.i_bayes = data["i_bayes"]
        return mf


if __name__ == '__main__':
    RATINGS_FILE = "../../data/MovieLens/ml-latest-small/ratings.csv"
    uir_data = UIRData.from_csv(RATINGS_FILE)

    mf = MatrixFactorization(k=20, max_iteration=20, batch_size=100)
    mf.fit(uir_data)
    # mf.save("../../models/ml-small-mf")

    r = mf.predict(uir_data.unique_values.users[0],
                   uir_data.unique_values.items[0])
    print(f"item 0, user 0: {r}")
