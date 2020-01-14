import numpy as np

from pyrec.inventory import Inventory
from pyrec.recommender import BaseRecommender


class RandomSimulator:
    def __init__(self, data: np.ndarray, recommender: BaseRecommender, inventory: Inventory):
        self.data = data
        self.rec = recommender
        self.inv = inventory

    def select_item(self, u, top=10):
        """
        Select random item from topN with probability corresponding to rating
        """
        items, r = self.rec.top_n(u, top)
        p = r / np.sum(r)
        choice = np.random.choice(top, p=p)
        return items[choice], r[choice]

    def run(self, iter=1000):
        users = np.unique(self.data[:, 0])

        ratings = []
        miss = 0
        for _ in range(iter):
            u = users[np.random.randint(len(users))]
            i, r = self.select_item(u)
            if not self.inv.is_empty(i):
                self.inv.remove_item(i)
                ratings.append(r)
            else:
                miss += 1
        sold_out = np.sum(self.inv.counts == 0)

        print(f"Avg rating: {sum(ratings) / len(ratings)}")
        print(f"Percent of misses: {miss / iter}")
        print(f"Sold out: {sold_out / len(self.inv.counts)}")


if __name__ == '__main__':
    from pyrec.data import UIRData
    from pyrec.recommender import MatrixFactorization

    RATINGS_FILE = "../data/MovieLens/ml-latest-small/ratings.csv"
    uir_data = UIRData.from_csv(RATINGS_FILE)

    mf = MatrixFactorization.load("../models/ml-small-mf")
    inv = Inventory(uir_data)
    sim = RandomSimulator(uir_data, mf, inv)
    sim.run()
