from typing import Union

import numpy as np

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
        print(f"All items in inventory: {self.start_size}")

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

    def percent_empty(self):
        return np.sum(self.counts == 0) / len(self.counts)

    def __bool__(self):
        return not np.any(self.counts)


if __name__ == '__main__':
    RATINGS_FILE = "../data/MovieLens/ml-latest-small/ratings.csv"

    data = UIRData.from_csv(RATINGS_FILE)
    item = data.unique_values.items[0]

    inv = Inventory(data)
    inv.remove_item(item)
    print(inv.is_empty(item))
    print(inv.item_count(item))
