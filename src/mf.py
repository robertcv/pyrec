import time
import numpy as np


def reduce_unique(data, col=0, min_ratings=50):
    u, c = np.unique(data[:, col], return_counts=True)
    select = np.isin(data[:, col], u[c > min_ratings])
    return data[select]


class MatrixFactorization:

    def __init__(self, k=20, max_iteration=50, batch_size=10000, test_r=0.3,
                 alpha=0.001, mi=0.0001):
        self.k = k
        self.max_iteration = max_iteration
        self.batch_size = batch_size
        self.test_r = test_r
        self.alpha = alpha
        self.mi = mi

        self.users = np.array([])
        self.items = np.array([])

        self.P = np.array([])
        self.Q = np.array([])
        self.u_bayes = np.array([])
        self.i_bayes = np.array([])

        self.u_avg = np.array([])
        self.i_avg = np.array([])
        self.g_avg = 0

    def fit(self, data):
        self.users, users_pos = np.unique(data[:, 0], return_inverse=True)
        self.items, items_pos = np.unique(data[:, 1], return_inverse=True)
        ratings = data[:, 2]
        N, M = len(self.users), len(self.items)
        print(f"Full size: ({N}, {M})")

        self.fit_avg(data)
        ratings -= self.g_avg

        train_size = int(len(data) * (1 - self.test_r))
        train_u = users_pos[:train_size]
        train_i = items_pos[:train_size]
        train_r = ratings[:train_size]
        train_select = np.arange(train_size)

        test_u = users_pos[train_size:]
        test_i = items_pos[train_size:]
        test_r = ratings[train_size:]

        self.P = np.random.normal(size=(N, self.k), scale=1/self.k)
        self.Q = np.random.normal(size=(M, self.k), scale=1/self.k)

        self.u_bayes = np.random.normal(size=(N,), scale=.1/self.k)
        self.i_bayes = np.random.normal(size=(M,), scale=.1/self.k)

        n_batches = np.clip(int(len(data) / self.batch_size), 1, len(data))
        print(f"Number of Batches: {n_batches}")

        for _ in range(self.max_iteration):
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
            train_e /= train_size

            # calculate error on test
            test_e = np.mean(np.abs(test_r - self._pred_vec(test_u, test_i)))
            iter_t = time.time() - start_t
            print(f"iter t: {iter_t:.5f}, train e: {train_e:.5f}, test e: {test_e:.5f}")

    def _pred_vec(self, u, i):
        return np.sum(self.P[u, :] * self.Q[i, :], axis=1) \
               + self.u_bayes[u] + self.i_bayes[i]

    def fit_avg(self, data):
        self.u_avg = np.zeros(len(self.users))
        for i, u in enumerate(self.users):
            self.u_avg[i] = np.mean(data[data[:, 0] == u, 2])

        self.i_avg = np.zeros(len(self.items))
        for j, i in enumerate(self.items):
            self.i_avg[j] = np.mean(data[data[:, 1] == i, 2])

        self.g_avg = np.mean(data[:, 2])
        print(f"Global avg: {self.g_avg}")

    def predict(self, user, item):
        u = np.where(self.users == user)[0]
        i = np.where(self.items == item)[0]
        if u and i:
            return self._pred_vec(u, i) + self.g_avg
        elif u:
            return self.u_avg[u[0]]
        elif i:
            return self.i_avg[i[0]]
        else:
            return self.g_avg

    def top_n(self, user, n):
        u = np.where(self.users == user)[0][0]
        m = len(self.items)
        pred = self._pred_vec(np.repeat(u, m), np.arange(m))
        top_n = np.argsort(pred)[-n:]
        return self.items[top_n], pred[top_n]

    def save(self, file_name):
        if not file_name.endswith(".npz"):
            file_name += ".npz"
        np.savez(file_name,
                 users=self.users, items=self.items,
                 P=self.P, Q=self.Q,
                 u_bayes=self.u_bayes, i_bayes=self.i_bayes)

    @staticmethod
    def load(file_name):
        if not file_name.endswith(".npz"):
            file_name += ".npz"
        data = np.load(file_name)
        mf = MatrixFactorization()
        mf.users = data["users"]
        mf.items = data["items"]
        mf.P = data["P"]
        mf.Q = data["Q"]
        mf.u_bayes = data["u_bayes"]
        mf.i_bayes = data["i_bayes"]


if __name__ == '__main__':
    import pandas as pd
    # RATINGS_FILE = "/home/robertcv/mag/data/MovieLens/ml-latest/ratings.csv"
    RATINGS_FILE = "/home/robertcv/mag/data/MovieLens/ml-latest-small/ratings.csv"

    # userId, movieId, rating
    df = pd.read_csv(RATINGS_FILE)
    # df = df.sort_values(by="timestamp")
    data = df.values[:, :-1]
    # good = data[:, 2] > 3
    # data = data[good]
    # data[:, 2] = 1
    data = reduce_unique(data, min_ratings=100)
    data = reduce_unique(data, col=1, min_ratings=100)
    np.random.shuffle(data)

    mf = MatrixFactorization(k=20, max_iteration=100, batch_size=100,
                             test_r=0.1)
    mf.fit(data)

    print(mf.top_n(1, 5))
