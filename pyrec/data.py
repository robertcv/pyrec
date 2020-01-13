from collections import namedtuple

import numpy as np
import pandas as pd


class UIRData:

    uir_type = namedtuple("_Data", ["users", "items", "ratings"])

    def __init__(self, users, items, ratings):
        self.raw_data = UIRData.uir_type(users=np.array(users),
                                         items=np.array(items),
                                         ratings=np.array(ratings,
                                                          dtype=np.float))
        self.uir_n = len(users)
        self.n, self.m = 0, 0

        self.unique_values = None  # type: UIRData.uir_type
        self.indexed_data = None  # type: UIRData.uir_type

        self.train_index = None  # type: np.ndarray
        self.validate_index = None  # type: np.ndarray
        self.test_indexes = None  # type: np.ndarray

        self.is_preprocessed = False

    @staticmethod
    def from_csv(file_name):
        df = pd.read_csv(file_name)
        return UIRData(df.values[:, 0], df.values[:, 1], df.values[:, 2])

    def preprocess(self):
        users, users_pos = np.unique(self.raw_data.users, return_inverse=True)
        items, items_pos = np.unique(self.raw_data.items, return_inverse=True)

        self.unique_values = UIRData.uir_type(users=users,
                                              items=items,
                                              ratings=None)

        self.n = len(self.unique_values.users)
        self.m = len(self.unique_values.items)

        self.indexed_data = UIRData.uir_type(users=users_pos,
                                             items=items_pos,
                                             ratings=None)
        self.is_preprocessed = True

    def reduce(self, items=False, min_ratings=50):
        col = 1 if items else 0
        u, c = np.unique(self.raw_data[col], return_counts=True)
        subset = np.isin(self.raw_data[col], u[c > min_ratings])
        self.raw_data = UIRData.uir_type(users=self.raw_data.users[subset],
                                         items=self.raw_data.items[subset],
                                         ratings=self.raw_data.ratings[subset])

    def split(self, train=0.6, validate=0.1, test=0.3, keep_order=False):
        train, validate, test = np.array([train, validate, test]) / \
                                (train + validate + test)

        indexes = np.arange(self.uir_n)
        if not keep_order:
            np.random.shuffle(indexes)

        self.train_index, self.validate_index, self.test_indexes = \
            np.split(indexes, [int(train * self.uir_n),
                               int((train + validate) * self.uir_n)])

    def train_data(self):
        return self.indexed_data.users[self.train_index], \
               self.indexed_data.items[self.train_index], \
               self.raw_data.ratings[self.train_index]

    def validate_data(self):
        return self.indexed_data.users[self.validate_index], \
               self.indexed_data.items[self.validate_index], \
               self.raw_data.ratings[self.validate_index]

    def test_data(self):
        return self.indexed_data.users[self.test_indexes], \
               self.indexed_data.items[self.test_indexes], \
               self.raw_data.ratings[self.test_indexes]
