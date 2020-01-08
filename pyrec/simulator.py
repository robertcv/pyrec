import numpy as np
from pyrec.inventory import Inventory
from pyrec.recommender import BaseRecommender
from pyrec.mf import MatrixFactorization


class RandomSimulator:
    def __init__(self, users, recommender: BaseRecommender, inventory: Inventory):
        self.users = list(users)
        self.rec = recommender
        self.inv = inventory

    def run(self):
        ratings = []
        empty = 0
        while not self.inv:
            u = self.users[np.random.randint(len(self.users))]
            i, r = self.rec.top_n(u, 1)
            i, r = i[0], r[0]
            if not self.inv.is_empty(i):
                self.inv.remove_item(i)
                ratings.append(r)
            else:
                empty += 1

        print(f"Avg rating: {sum(ratings) / len(ratings)}")
        print(f"Recommended empty: {empty}")


if __name__ == '__main__':
    import pandas as pd
    RATINGS_FILE = "/home/robertcv/mag/data/MovieLens/ml-latest-small/ratings.csv"

    df = pd.read_csv(RATINGS_FILE)
    data = df.values[:, :-1]

    mf = MatrixFactorization.load("../models/ml-small-mf")
    inv = Inventory(data)
    users = mf.users[:10]
    sim = RandomSimulator(users, mf, inv)
    sim.run()
