from typing import List

import numpy as np
import matplotlib.pyplot as plt

from pyrec.data import UIRData
from pyrec.inventory import Inventory
from pyrec.recommender import BaseRecommender


class BaseSimulator:
    def __init__(self, name, data: UIRData, rec: BaseRecommender,
                 inv: Inventory):
        self.name = name
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
        sold_items = []

        for _ in range(iter):
            u = self.select_user()
            i, r = self.select_item(u)

            if u in self.data.hier_ratings and i in self.data.hier_ratings[u]:
                ratings_diff.append((r - self.data.hier_ratings[u][i]) ** 2)

            if not self.inv.is_empty(i):
                self.inv.remove_item(i)
                sold_items.append(self.inv.percent_sold() * 100)
            else:
                sold_items.append(sold_items[-1])
            empty_items.append(self.inv.percent_empty() * 100)

        print(f"Average rmse: {np.sqrt(sum(ratings_diff) / len(ratings_diff))} for {len(ratings_diff)} ratings")
        print(f"Percent empty items: {empty_items[-1]}")
        print(f"Percent sold items: {sold_items[-1]}")

        self.sim_data = {
            "empty_items": empty_items,
            "sold_items": sold_items
        }

    def plot(self, save_file=None):
        empty_items = np.array(self.sim_data["empty_items"])
        sold_items = np.array(self.sim_data["sold_items"])

        fig, ax = plt.subplots()
        ax.plot(empty_items)
        ax.plot(sold_items)

        if save_file is not None:
            fig.savefig(save_file)
        plt.show()

    @staticmethod
    def multi_plot(simulations: List['BaseSimulator'], data="empty_items",
                   save_file=None):
        fig, ax = plt.subplots()

        for sim in simulations:
            ax.plot(sim.sim_data[data], label=sim.name)

        ax.legend()

        if save_file is not None:
            fig.savefig(save_file)
        plt.show()
