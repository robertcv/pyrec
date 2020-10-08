from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyrec.data import UIRData


class Inventory:
    def __init__(self, data: Union[UIRData, np.ndarray]):
        if isinstance(data, UIRData):
            unique, unique_counts = np.unique(data.raw_data.items,
                                              return_counts=True)
        elif isinstance(data, np.ndarray):
            unique, unique_counts = np.array(data[:, 0]), \
                                    np.array(data[:, 1], dtype=int)
        else:
            raise ValueError("unsupported datatype")

        self.items = unique
        self.item2i = {item: i for i, item in enumerate(self.items)}
        self.counts = unique_counts
        self.start_size = np.sum(self.counts)
        self.name = "inv"

    @staticmethod
    def from_csv(file_name: str) -> 'Inventory':
        """
        Load csv data from file_name. The first columns is assumed
        to be item ids and second inventory count.
        :param file_name: location of the csv file
        :return: a new Inventory object initialized with data from csv
        """
        df = pd.read_csv(file_name)
        return Inventory(df.values[:, :2])

    def __repr__(self):
        return f"Inv ({self.current_count()} all, {len(self.counts)} items)"

    def copy(self) -> 'Inventory':
        inv = type(self)(np.zeros((1, 2)))
        inv.items = np.array(self.items)
        inv.item2i = self.item2i.copy()
        inv.counts = np.array(self.counts)
        inv.start_size = self.start_size
        return inv

    def reduce(self, p=0.1):
        self.counts = (self.counts * p).astype(int)
        self.counts[self.counts == 0] = 1
        self.start_size = np.sum(self.counts)

    def remove_item(self, item):
        i = self.item2i[item]
        self.counts[i] = max(0, self.counts[i] - 1)

    def is_empty(self, item):
        return not self.item_count(item)

    def item_count(self, item):
        return self.counts[self.item2i[item]]

    def current_count(self):
        return np.sum(self.counts)

    def percent_left(self):
        return self.current_count() / self.start_size

    def percent_sold(self):
        return 1 - self.percent_left()

    def percent_empty(self):
        return np.sum(self.counts == 0) / len(self.counts)

    def __bool__(self):
        return not np.any(self.counts)

    def plot_dist(self):
        plt.hist(self.counts, color='blue', edgecolor='black', bins=100)
        plt.show()


class UniformInventory(Inventory):
    def __init__(self, data: Union[UIRData, np.ndarray]):
        super().__init__(data)
        n = int(self.start_size / len(self.items))
        self.counts = np.ones(len(self.items), dtype=int) * n
        self.start_size = len(self.items) * n
        self.name = "uinv"

    def copy(self):
        inv = super().copy()
        inv.counts = np.array(self.counts)
        inv.start_size = self.start_size
        return inv


class RandomInventory(Inventory):
    def __init__(self, data: Union[UIRData, np.ndarray]):
        super().__init__(data)
        np.random.seed(None)
        np.random.shuffle(self.counts)
        self.name = "rinv"

    def copy(self):
        inv = super().copy()
        np.random.seed(None)
        np.random.shuffle(inv.counts)
        return inv


if __name__ == '__main__':
    RATINGS_FILE = "../data/MovieLens/ml-latest-small/ratings.csv"

    data = UIRData.from_csv(RATINGS_FILE)
    item = data.unique_values.items[0]

    inv = Inventory(data)
    print(inv)
    inv.remove_item(item)
    print(inv.is_empty(item))
    print(inv.item_count(item))
    inv.plot_dist()
