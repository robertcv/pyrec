from typing import Optional, NamedTuple, Dict, Any, Set
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd


uir_type = NamedTuple("uir_type", [("users", np.ndarray),
                                   ("items", np.ndarray),
                                   ("ratings", Optional[np.ndarray])])


class UIRData:
    """
    Class encapsulating (user, item, rating) style data and metadata for
    recommender systems.
    """
    def __init__(self,
                 users: np.ndarray, items: np.ndarray, ratings: np.ndarray,
                 auto_r=False, auto_prep=True, auto_s=True, auto_ptrain=True,
                 auto_ptest=True):

        self.raw_data = uir_type(users=np.array(users),
                                 items=np.array(items),
                                 ratings=np.array(ratings, dtype=np.float))
        self.uir_n = len(users)
        self.n, self.m = 0, 0

        self.unique_values = None  # type: Optional[uir_type]
        self.indexed_data = None  # type: Optional[uir_type]

        self.user2index = {}
        self.item2index = {}

        self._train_data = None  # type: Optional[uir_type]
        self._validation_data = None  # type: Optional[uir_type]
        self._test_data = None  # type: Optional[uir_type]
        self._hier_test_ratings = None  # type: Optional[Dict[Any, Dict[Any, float]]]
        self._hier_bought = None  # type: Optional[Dict[Any, Set]]

        self._user_avg = None  # type: Optional[np.ndarray]
        self._item_avg = None  # type: Optional[np.ndarray]
        self._global_avg = 0

        if auto_r:
            self.reduce()
            self.reduce(items=True)

        if auto_prep:
            self.preprocess()

        if auto_s:
            self.split()

        if auto_ptrain:
            self.preprocess_train()

        if auto_ptest:
            self.preprocess_test()

    def __repr__(self):
        return f"URI ({self.uir_n} rows, {self.n} users, {self.m} items)"

    @staticmethod
    def from_csv(file_name: str, **kwargs) -> 'UIRData':
        """
        Load csv data from file_name. The first three columns are assumed
        to be user ids, item ids and ratings.
        :param file_name: location of the csv file
        :return: a new URIData object initialized with data from csv
        """
        df = pd.read_csv(file_name)
        return UIRData(df.values[:, 0], df.values[:, 1], df.values[:, 2],
                       **kwargs)

    def copy(self):
        return UIRData(self.raw_data.users,
                       self.raw_data.items,
                       self.raw_data.ratings)

    def preprocess(self):
        """
        Prepare some metadata for fester data manipulation.
        """
        users, users_pos = np.unique(self.raw_data.users, return_inverse=True)
        items, items_pos = np.unique(self.raw_data.items, return_inverse=True)

        self.unique_values = uir_type(users=users,
                                      items=items,
                                      ratings=None)

        self.n = len(self.unique_values.users)
        self.m = len(self.unique_values.items)

        self.indexed_data = uir_type(users=users_pos,
                                     items=items_pos,
                                     ratings=None)

        self.user2index = {user: i for i, user in enumerate(users)}
        self.item2index = {item: i for i, item in enumerate(items)}

    def preprocess_train(self):
        """
        Calculate average ratings for user, movies and globally for the
        training dataset. For users/items not in traning dataset we use nan.
        """
        df = pd.DataFrame(data={"users": self.train_data.users,
                                "items": self.train_data.items,
                                "ratings": self.train_data.ratings})

        users_df = pd.DataFrame(data={"users": np.arange(self.n)})
        users_avg_df = df.groupby("users", as_index=False).mean()
        self._user_avg = users_df.merge(users_avg_df, how="left")["ratings"].values

        items_df = pd.DataFrame(data={"items": np.arange(self.m)})
        items_avg_df = df.groupby("items", as_index=False).mean()
        self._item_avg = items_df.merge(items_avg_df, how="left")["ratings"].values

        self._global_avg = np.mean(self.train_data.ratings)

    def preprocess_test(self):
        """
        Save test data ratings and items already 'bought' from train and
        validation data into dictionaries for faster access.
        """
        self._hier_test_ratings = defaultdict(dict)
        for u, i, r in zip(*self.test_data):
            self._hier_test_ratings[self.unique_values.users[u]][self.unique_values.items[i]] = r

        self._hier_bought = defaultdict(set)
        for u, i, _ in zip(*self.train_data):
            self._hier_bought[self.unique_values.users[u]].add(self.unique_values.items[i])
        for u, i, _ in zip(*self.validation_data):
            self._hier_bought[self.unique_values.users[u]].add(self.unique_values.items[i])

    def reduce(self, items=False, min_ratings=50):
        """
        Remove users/items that don't have at least min_ratings
        number of ratings.
        """
        col = 1 if items else 0
        u, c = np.unique(self.raw_data[col], return_counts=True)
        subset = np.isin(self.raw_data[col], u[c > min_ratings])
        self.raw_data = uir_type(users=self.raw_data.users[subset],
                                 items=self.raw_data.items[subset],
                                 ratings=self.raw_data.ratings[subset])

    def split(self, train=0.6, validation=0.1, keep_order=False):
        """
        Prepare the split of data into train, validation and test sets.
        If keep_order then data is split consecutively.
        :param train: proportion of train data
        :param validation: proportion of validation data
        :param keep_order: keep order of input data
        """
        self.__needs_preprocess()

        indexes = np.arange(self.uir_n)
        if not keep_order:
            np.random.shuffle(indexes)

        train_index, validation_index, test_indexes = \
            np.split(indexes, [int(train * self.uir_n),
                               int((train + validation) * self.uir_n)])

        self._train_data = uir_type(
            users=self.indexed_data.users[train_index],
            items=self.indexed_data.items[train_index],
            ratings=self.raw_data.ratings[train_index]
        )
        self._validation_data = uir_type(
            users=self.indexed_data.users[validation_index],
            items=self.indexed_data.items[validation_index],
            ratings=self.raw_data.ratings[validation_index]
        )
        self._test_data = uir_type(
            users=self.indexed_data.users[test_indexes],
            items=self.indexed_data.items[test_indexes],
            ratings=self.raw_data.ratings[test_indexes]
        )

    def __needs_preprocess(self):
        if self.indexed_data is None:
            self.preprocess()

    def __needs_split(self):
        if self._train_data is None:
            self.split()

    def __needs_preprocess_train(self):
        if self._user_avg is None:
            self.preprocess_train()

    def __needs_preprocess_test(self):
        if self._hier_test_ratings is None:
            self.preprocess_test()

    @property
    def index2user(self):
        self.__needs_preprocess()
        return self.unique_values.users

    @property
    def index2item(self):
        self.__needs_preprocess()
        return self.unique_values.items

    @property
    def train_data(self) -> uir_type:
        self.__needs_split()
        return self._train_data

    @property
    def validation_data(self) -> uir_type:
        self.__needs_split()
        return self._validation_data

    @property
    def test_data(self) -> uir_type:
        self.__needs_split()
        return self._test_data

    @property
    def user_avg(self) -> np.ndarray:
        self.__needs_preprocess_train()
        return self._user_avg

    @property
    def item_avg(self) -> np.ndarray:
        self.__needs_preprocess_train()
        return self._item_avg

    @property
    def global_avg(self) -> float:
        self.__needs_preprocess_train()
        return self._global_avg

    @property
    def hier_test_ratings(self) -> Dict[Any, Dict[Any, float]]:
        """by value"""
        self.__needs_preprocess_test()
        return self._hier_test_ratings

    @property
    def hier_bought(self) -> Dict[Any, Set]:
        """by values"""
        self.__needs_preprocess_test()
        return deepcopy(self._hier_bought)


if __name__ == '__main__':
    movie_lens = "../data/MovieLens/ml-latest-small/ratings.csv"
    uir_data = UIRData.from_csv(movie_lens)
    print(uir_data)
    print(uir_data.train_data)
