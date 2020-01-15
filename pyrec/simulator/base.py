import numpy as np

from pyrec.data import UIRData
from pyrec.inventory import Inventory
from pyrec.recommender import BaseRecommender


class BaseSimulator:
    def __init__(self, data: UIRData, rec: BaseRecommender, inv: Inventory):
        self.data = data
        self.rec = rec
        self.inv = inv

        self.sim_data = {}

    def select_item(self, u):
        """Return selected item and it's rating."""
        raise NotImplementedError

    def select_user(self):
        """Return selected user."""
        raise NotImplementedError

    def run(self, iter=1000):
        ratings_diff = []
        empty_items = []
        current_count = []

        for _ in range(iter):
            u = self.select_user()
            i, r = self.select_item(u)

            if u in self.data.hier_ratings and i in self.data.hier_ratings[u]:
                ratings_diff.append((r - self.data.hier_ratings[u][i]) ** 2)

            if not self.inv.is_empty(i):
                self.inv.remove_item(i)
                current_count.append(self.inv.percent_left())
            else:
                current_count.append(current_count[-1])
            empty_items.append(np.sum(self.inv.counts == 0) / len(self.inv.counts))

        print(f"Average rmse: {np.sqrt(sum(ratings_diff) / len(ratings_diff))} for {len(ratings_diff)} ratings")
        print(f"Percent empty items: {empty_items[-1]}")
        print(f"Percent items count: {current_count[-1]}")

        self.sim_data = {
            "empty_items": empty_items,
            "current_count": current_count
        }

    def plot(self):
        pass

