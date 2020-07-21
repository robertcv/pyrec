import time

import numpy as np

from pyrec.recs.mf import MatrixFactorization
from pyrec.data import UIRData
from pyrec.inventory import Inventory


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class CapMF(MatrixFactorization):
    def __init__(self, inv: Inventory, walpha=0.5,
                 k=20, max_iteration=1000, batch_size=100,
                 alpha=0.01, mi=0.001, verbose=True, seed=None, **kwargs):
        super().__init__(k=k, max_iteration=max_iteration,
                         batch_size=batch_size, alpha=alpha, mi=mi,
                         verbose=verbose, seed=seed, **kwargs)
        self.inv = inv
        self.walpha = walpha

        self.cache_n = 0
        self.eu = None
        self.p_i = None

    def _init_matrix(self):
        super()._init_matrix()
        self.u_bayes *= 0
        self.i_bayes *= 0

    def fit(self, data: UIRData):
        self.data = data
        np.random.seed(self.seed)
        self._init_matrix()

        train_u = data.train_data.users
        train_i = data.train_data.items
        # ratings have to be normalised for capacity loss
        train_r = data.train_data.ratings / data.train_data.ratings.max()

        train_len = len(train_u)
        train_select = np.arange(train_len)

        n_batches = np.clip(int(train_len / self.batch_size), 1, train_len)
        self._print_verbose(f"Number of Batches: {n_batches}")

        validation_u = data.validation_data.users
        validation_i = data.validation_data.items
        validation_r = data.validation_data.ratings / data.validation_data.ratings.max()

        self.p_i = np.zeros(self.data.n)
        for u in np.arange(self.data.n):
            self.p_i[u] = len(np.unique(self.data.indexed_data.items[self.data.indexed_data.users == u])) / self.data.m

        L_i = {}
        for u in np.unique(train_u):
            L_i[u] = np.unique(train_i[train_u == u])

        Ra = {}
        for i in np.unique(train_i):
            Ra[i] = np.unique(train_u[train_i == i])

        r_ij = np.zeros((self.data.n, self.data.m))
        r_ij[train_u, train_i] = train_r
        start_t = time.time()
        for iteration in range(self.max_iteration):
            index = 0
            np.random.shuffle(train_select)
            for i, j in zip(train_u[train_select], train_i[train_select]):
                # update user
                user_gradient = -self.walpha * np.sum(2 * (r_ij[i, L_i[i]] - self._pred_vec(i, L_i[i]))[:, np.newaxis] * self.Q[L_i[i], :], axis=0)
                user_gradient += 2 * self.mi * self.P[i, :]

                cl = self.inv.counts - self._expected_usage()
                _r_ij = self._pred_vec(i, np.arange(self.data.m))
                tmp = sigmoid(-cl) * self.p_i[i] * sigmoid(_r_ij) * sigmoid(-_r_ij)
                user_gradient += ((1 - self.walpha) / self.data.m) * np.sum(tmp[:, np.newaxis] * self.Q, axis=0)

                self.P[i, :] = self.P[i, :] - self.alpha * user_gradient

                # update item
                item_gradient = -self.walpha * np.sum(2 * (r_ij[Ra[j], j] - self._pred_vec(Ra[j], j))[:, np.newaxis] * self.P[Ra[j], :], axis=0)
                item_gradient += 2 * self.mi * self.Q[j, :]

                cl = self.inv.counts[j] - self._expected_usage()[j]
                _r_ij = self._pred_vec(np.arange(self.data.n), j)
                tmp = self.p_i * sigmoid(_r_ij) * sigmoid(-_r_ij)
                item_gradient += ((1 - self.walpha) / self.data.m) * sigmoid(-cl) * np.sum(tmp[:, np.newaxis] * self.P, axis=0)

                self.Q[i, :] = self.Q[i, :] - self.alpha * item_gradient

                if index % 100 == 0:
                    ame = np.mean(np.abs(validation_r - self._pred_vec(validation_u, validation_i)))
                    c_loss = np.mean(self.inv.counts[validation_i] - np.sum(self.p_i[validation_u] * sigmoid(self._pred_vec(validation_u, validation_i))))

                    iter_t = time.time() - start_t
                    self._print_verbose(f"iter t: {iter_t:.3f} ame: {ame:.3f} c_loss: {c_loss:.3f}")
                    start_t = time.time()
                index += 1

    def _pred_vec(self, u, i):
        if isinstance(u, np.ndarray) or isinstance(i, np.ndarray):
            return np.sum(self.P[u, :] * self.Q[i, :], axis=1)
        return np.sum(self.P[u, :] * self.Q[i, :])

    def _expected_usage(self):
        if self.cache_n % 100 == 0:
            self.eu = np.sum(self.p_i[:, np.newaxis] * sigmoid(np.dot(self.P, self.Q.T)), axis=0)
        self.cache_n += 1
        return self.eu


if __name__ == '__main__':
    RATINGS_FILE = "../../data/MovieLens/ml-latest-small/ratings.csv"
    uir_data = UIRData.from_csv(RATINGS_FILE)
    inv = Inventory(uir_data)

    mf = CapMF(inv)
    mf.fit(uir_data)
    # mf.save("../../models/ml-small-mf")

    r = mf.predict(uir_data.unique_values.users[0],
                   uir_data.unique_values.items[0])
    print(f"item 0, user 0: {r}")
