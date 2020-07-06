import time
from typing import Tuple

import numpy as np

from pyrec.recs.mf import MatrixFactorization
from pyrec.data import UIRData, uir_type


class BPR(MatrixFactorization):
    def __init__(self, k=20, max_iteration=10_000, batch_size=100,
                 alpha=0.01, mi=0.001, verbose=True, seed=None, **kwargs):
        super().__init__(k=k, max_iteration=max_iteration,
                         batch_size=batch_size, alpha=alpha, mi=mi,
                         verbose=verbose, seed=seed, **kwargs)

    def fit(self, data: UIRData):
        self.data = data
        np.random.seed(self.seed)
        self._init_matrix()

        train_i = data.train_data.items
        unique_t_u, user_t_r = self._preprocess(data.train_data)
        validation_i = data.validation_data.items
        unique_v_u, user_v_r = self._preprocess(data.validation_data)

        def _validation_loss():
            bpr_opt = []
            for u in unique_v_u:
                i = validation_i[user_v_r[u]['pos']]
                j = validation_i[user_v_r[u]['neg']]
                x_ui = np.sum(self.P[u, :] * self.Q[i, :]) + self.i_bayes[i]
                x_uj = np.sum(self.P[u, :] * self.Q[j, :]) + self.i_bayes[j]
                x_uij = x_ui[:, None] - x_uj
                sig = 1 / (1 + np.exp(-x_uij))
                bpr_opt.append(np.sum(np.log(sig)))
            return np.sum(bpr_opt)

        last_loss = _validation_loss()

        for iteration in range(self.max_iteration):
            start_t = time.time()
            u = np.random.choice(unique_t_u, size=self.batch_size)
            i = np.array([train_i[np.random.choice(user_t_r[_u]['pos'])] for _u in u])
            j = np.array([train_i[np.random.choice(user_t_r[_u]['neg'])] for _u in u])
            self._update(u, i, j)

            if iteration % 50 == 0 and iteration != 0:
                validation_loss = _validation_loss()

                iter_t = (time.time() - start_t)
                self._print_verbose(f"iter t: {iter_t:.5f}, loss: {validation_loss:.5f}")

                if last_loss > validation_loss:
                    self._print_verbose(f"Ending after iteration {iteration}")
                    break
                else:
                    last_loss = validation_loss

    def _update(self, u, i, j):
        x_uij = np.sum(self.P[u, :] * (self.Q[i, :] - self.Q[j, :]), axis=1) + self.i_bayes[i] - self.i_bayes[j]
        z = (1 / (np.exp(x_uij) + 1))

        self.P[u, :] += self.alpha * (z[:, np.newaxis] * (self.Q[i, :] - self.Q[j, :]) + self.mi * self.P[u, :])
        self.Q[i, :] += self.alpha * (z[:, np.newaxis] * self.P[u, :] + self.mi * self.Q[i, :])
        self.Q[j, :] += self.alpha * (z[:, np.newaxis] * -self.P[u, :] + self.mi * self.Q[j, :])

        self.i_bayes[i] += self.alpha * (z + self.mi * self.i_bayes[i])
        self.i_bayes[j] += self.alpha * (-z + self.mi * self.i_bayes[j])

    def _pred_vec(self, u, i):
        return np.sum(self.P[u, :] * self.Q[i, :], axis=1) + self.i_bayes[i]

    def _preprocess(self, data: uir_type) -> Tuple[np.ndarray, dict]:
        unique_u = np.unique(data.users)
        processed_data = {}
        for u in unique_u:
            u_bool = data.users == u
            r_mean = 3.5
            neg = np.where(u_bool & (data.ratings <= r_mean))[0]
            if neg.size < 1:
                continue
            pos = np.where(u_bool & (data.ratings > r_mean))[0]
            if pos.size < 1:
                continue
            processed_data[u] = {'pos': pos, 'neg': neg}
        unique_u = np.array(list(processed_data.keys()))
        return unique_u, processed_data


class UnbiasedBPR(BPR):
    def _update(self, u, i, j):
        x_uij = np.sum(self.P[u, :] * (self.Q[i, :] - self.Q[j, :]), axis=1)
        z = (1 / (np.exp(x_uij) + 1))
        self.P[u, :] += self.alpha * (z[:, np.newaxis] * (self.Q[i, :] - self.Q[j, :]) + self.mi * self.P[u, :])
        self.Q[i, :] += self.alpha * (z[:, np.newaxis] * self.P[u, :] + self.mi * self.Q[i, :])
        self.Q[j, :] += self.alpha * (z[:, np.newaxis] * -self.P[u, :] + self.mi * self.Q[j, :])

    def _init_matrix(self):
        super()._init_matrix()
        self.u_bayes *= 0
        self.i_bayes *= 0


if __name__ == '__main__':
    RATINGS_FILE = "../../data/MovieLens/ml-latest-small/ratings.csv"
    uir_data = UIRData.from_csv(RATINGS_FILE)

    bpr = BPR(k=20, alpha=0.01, mi=0.001, batch_size=500)
    bpr.fit(uir_data)

    r = bpr.predict(uir_data.unique_values.users[0],
                    uir_data.unique_values.items[0])
    print(f"item 0, user 0: {r}")
