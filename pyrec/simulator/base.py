from typing import List
from copy import deepcopy
from multiprocessing import Process, Manager, Queue

import numpy as np
import matplotlib.pyplot as plt

from pyrec.data import UIRData
from pyrec.inventory import Inventory
from pyrec.recommender import BaseRecommender


class BaseSimulator:
    def __init__(self, name, data: UIRData, rec: BaseRecommender,
                 inv: Inventory, verbose=True):
        self.name = name
        self.data = data
        self.rec = rec
        self.inv = inv
        self.verbose = verbose

        self.sim_data = {}

    def select_item(self, u):
        """Return selected item and it's rating."""
        raise NotImplementedError

    def select_user(self):
        """Return selected user."""
        raise NotImplementedError

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
        sold_items = []

        _n = n // 10

        for _i in range(n):
            if _i % _n == 0:
                self._print_verbose(_i / n)

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

        self._print_verbose(1)

        self.sim_data = {
            "empty_items": empty_items,
            "sold_items": sold_items,
            "rmse": np.sqrt(sum(ratings_diff) / len(ratings_diff)),
            "rmse_number": len(ratings_diff),
        }
        return deepcopy(self.sim_data)

    def plot(self, save_file=None):
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

    @staticmethod
    def multi_plot(simulations: List['BaseSimulator'], data="empty_items",
                   save_file=None):
        fig, ax = plt.subplots()

        for sim in simulations:
            ax.plot(sim.sim_data[data], label=sim.name)

        ax.legend()
        ax.set_xlabel('iteration')
        ax.set_ylabel('percent')
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
