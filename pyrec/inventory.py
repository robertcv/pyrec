from typing import Union

import numpy as np
import matplotlib.pyplot as plt

from pyrec.data import UIRData


class Inventory:
    def __init__(self, data: Union[UIRData, np.ndarray]):
        if isinstance(data, UIRData):
            unique, unique_counts = np.unique(data.raw_data.items,
                                              return_counts=True)
        else:
            unique, unique_counts = np.unique(data, return_counts=True)

        self.items = unique
        self.item2i = {item: i for i, item in enumerate(self.items)}
        self.counts = unique_counts
        self.start_size = np.sum(self.counts)

    def __repr__(self):
        return f"Inv ({self.current_count()} all, {len(self.counts)} items)"

    def copy(self) -> 'Inventory':
        inv = Inventory(np.array([]))
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
        print(f"All items in inventory: {self.start_size}")


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
