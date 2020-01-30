import numpy as np

from pyrec.simulator import BaseSimulator


class RandomSimulator(BaseSimulator):
    def select_item(self, user):
        random_test_i = np.random.randint(len(self.data.test_data.items))
        i = self.data.test_data.items[random_test_i]
        item = self.data.unique_values.items[i]
        rating = self.rec.predict(user, item)
        return item, rating

    def select_user(self):
        random_test_u = np.random.randint(len(self.data.test_data.users))
        u = self.data.test_data.users[random_test_u]
        return self.data.unique_values.users[u]


class RandomFromTopNSimulator(RandomSimulator):
    def __init__(self, name, data, rec, inv, n=5):
        super().__init__(name, data, rec, inv)
        self.n = n

    def select_item(self, user):
        items, ratings = self.top_n(user, self.n)
        random_index = np.random.randint(self.n)
        return items[random_index], ratings[random_index]

    def top_n(self, user, n):
        not_bought = self.user_has_not_bought(user)

        pred = self.rec.predict_user(user)[not_bought]
        arg_sort = np.argsort(pred)[::-1]
        pred = pred[arg_sort]
        items = self.data.unique_values.items[not_bought][arg_sort]

        return self._top_n_multiple(items, pred, n)

    @staticmethod
    def _top_n_multiple(items, pred, n):
        top_items, top_pred = np.array(items[:n]), np.array(pred[:n])

        if pred[n - 1] == pred[n]:
            # if the last in top ratings and the next rating are the same
            # we must randomly select new such items
            same_pred = pred[n]
            same_size = np.sum(pred[:n] == same_pred)
            same_indexes = np.where(pred == same_pred)[0]
            new_items = np.random.choice(same_indexes, same_size, replace=False)
            top_items[-same_size:] = items[new_items]

        return top_items, top_pred


if __name__ == '__main__':
    from pyrec.data import UIRData
    from pyrec.recommender import MatrixFactorization
    from pyrec.inventory import Inventory

    RATINGS_FILE = "../../data/MovieLens/ml-latest-small/ratings.csv"
    uir_data = UIRData.from_csv(RATINGS_FILE)

    mf = MatrixFactorization.load("../../models/ml-small-mf")
    mf.data = uir_data
    inv = Inventory(uir_data)
    sim = RandomSimulator("rand", uir_data, mf, inv)
    sim.run()
