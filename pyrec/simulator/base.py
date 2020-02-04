from typing import List
from copy import deepcopy
from multiprocessing import Process, Manager, Queue

import numpy as np
from scipy.signal import convolve
import matplotlib.pyplot as plt

from pyrec.data import UIRData
from pyrec.inventory import Inventory
from pyrec.recommender import BaseRecommender


class BaseSimulator:
    def __init__(self, name, data: UIRData, rec: BaseRecommender,
                 inv: Inventory, verbose=True):
        self.name = name
        self.data = data
        self.bought = data.hier_bought
        self.test_ratings = data.hier_test_ratings
        self.rec = rec
        self.inv = inv
        self.verbose = verbose

        self.sim_data = {}

    def select_item(self, user):
        """Return selected item and it's rating."""
        raise NotImplementedError

    def select_user(self):
        """Return selected user."""
        raise NotImplementedError

    def user_has_not_bought(self, user) -> np.ndarray:
        """
        Return an array of booleans if item has previously been bought by user.
        """
        bought = np.zeros(len(self.data.unique_values.items))
        for item in self.bought[user]:
            bought[self.data.item2index[item]] = 1
        return ~(bought.astype(bool))

    def user_bought_item(self, user, item):
        """Note that item has been bought by user."""
        self.bought[user].add(item)

    def _print_verbose(self, per):
        if self.verbose:
            if per == 0:
                print(f"{self.name}: start")
            elif per == 1:
                print(f"{self.name}: finish")
            else:
                print(f"{self.name}: {int(per * 100)}%")

    def run(self, n=1000):
        ratings_diff = []
        empty_items = []
        sold_items = [0]
        not_sold_items = [0]
        test_ratings = [0]
        predicted_ratings = []

        _n = n // 10

        for _i in range(n):
            if _i % _n == 0:
                self._print_verbose(_i / n)

            user = self.select_user()
            item, r = self.select_item(user)

            predicted_ratings.append(r)
            if user in self.test_ratings and item in self.test_ratings[user]:
                test_ratings.append(self.test_ratings[user][item])
                ratings_diff.append((r - self.test_ratings[user][item]) ** 2)
            else:
                test_ratings.append(test_ratings[-1])

            if not self.inv.is_empty(item):
                self.inv.remove_item(item)
                self.user_bought_item(user, item)
                sold_items.append(self.inv.percent_sold() * 100)
                not_sold_items.append(not_sold_items[-1])
            else:
                sold_items.append(sold_items[-1])
                not_sold_items.append(not_sold_items[-1] + 1)
            empty_items.append(self.inv.percent_empty() * 100)

        self._print_verbose(1)

        try:
            rmse = np.sqrt(sum(ratings_diff) / len(ratings_diff))
        except ZeroDivisionError:
            rmse = self.data.raw_data.ratings.max() - \
                   self.data.raw_data.ratings.min()

        self.sim_data = {
            "empty_items": empty_items,
            "sold_items": sold_items[1:],
            "test_ratings": test_ratings[1:],
            "predicted_ratings": predicted_ratings,
            "not_sold_items": not_sold_items[1:],
            "rmse": rmse,
            "rmse_number": len(ratings_diff),
        }
        return deepcopy(self.sim_data)

    def rmse(self):
        print(f"{self.name}: {self.sim_data['rmse']:.2f} for {self.sim_data['rmse_number']} ratings")

    def plot_items(self, save_file=None):
        empty_items = np.array(self.sim_data["empty_items"])
        sold_items = np.array(self.sim_data["sold_items"])

        fig, ax = plt.subplots()
        ax.plot(empty_items, label="empty items")
        ax.plot(sold_items, label="items sold")
        ax.legend()
        ax.set_xlabel('iteration')
        ax.set_ylabel('percent')
        ax.set_title(f'Inventory change throughout {self.name} simulation')

        if save_file is not None:
            fig.savefig(save_file)
        else:
            plt.show()

    def plot_ratings(self, save_file=None, mean_size=50):
        test_ratings = np.array(self.sim_data["test_ratings"])
        predicted_ratings = np.array(self.sim_data["predicted_ratings"])

        b = np.ones(mean_size) / mean_size
        b2 = np.ones(mean_size * 2) / (mean_size * 2)
        test_ratings = convolve(test_ratings, b, mode="same")
        predicted_ratings = convolve(predicted_ratings, b2, mode="same")

        fig, ax = plt.subplots()
        ax.plot(test_ratings, label="test rating")
        ax.plot(predicted_ratings, label="predicted rating")
        ax.legend()
        ax.set_xlabel('iteration')
        ax.set_ylabel('rating')
        ax.set_title(f'Change of rating for item in {self.name} simulation')

        if save_file is not None:
            fig.savefig(save_file)
        else:
            plt.show()

    @staticmethod
    def multi_plot(simulations: List['BaseSimulator'], data="empty_items",
                   save_file=None):
        fig, ax = plt.subplots()

        for sim in simulations:
            ax.plot(sim.sim_data[data], label=sim.name)

        ax.legend()
        ax.set_xlabel('iteration')
        ax.set_title(f'Change of {data} throughout the simulations')

        if save_file is not None:
            fig.savefig(save_file)
        else:
            plt.show()


class MultiSimulator:
    def __init__(self, n=1000):
        self.sims = []
        self.n = n

    def set_sims(self, sims):
        self.sims = sims

    def run_parallel(self):
        jobs = []
        q = Manager().Queue()
        for i in range(len(self.sims)):
            p = Process(target=self.__run_sim,
                        args=(i, self.sims[i], self.n, q))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        while not q.empty():
            i, sim_data = q.get()
            self.sims[i].sim_data = sim_data
            self.sims[i].name += f" (RMSE={self.sims[i].sim_data['rmse']:.2f})"

    @staticmethod
    def __run_sim(i: int, sim: BaseSimulator, n: int, q: Queue):
        sim_data = sim.run(n)
        q.put((i, sim_data))
