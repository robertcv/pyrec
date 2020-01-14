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
        ratings = []
        empty_items = []
        current_count = []

        for _ in range(iter):
            u = self.select_user()
            i, r = self.select_item(u)
            if not self.inv.is_empty(i):
                self.inv.remove_item(i)
                ratings.append(r)
                current_count.append(self.inv.percent_left())
            else:
                ratings.append(ratings[-1])
                current_count.append(current_count[-1])
            empty_items.append(np.sum(self.inv.counts == 0) / len(self.inv.counts))

        print(f"Average rating: {sum(ratings) / iter}")
        print(f"Percent empty items: {empty_items[-1]}")
        print(f"Percent items count: {current_count[-1]}")

        self.sim_data = {
            "ratings": ratings,
            "empty_items": empty_items,
            "current_count": current_count
        }

    def plot(self):
        pass

