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
        items, ratings = self.rec.top_n(user, n=self.n)
        random_index = np.random.randint(len(items))
        return items[random_index], ratings[random_index]


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
