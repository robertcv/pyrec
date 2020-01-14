from typing import Optional, NamedTuple

import numpy as np
import pandas as pd


class UIRData:
    """
    Class encapsulating (user, item, rating) style data and metadata for
    recommender systems.
    """
    uir_type = NamedTuple("uir_type", [("users", np.ndarray),
                                       ("items", np.ndarray),
                                       ("ratings", Optional[np.ndarray])])

    def __init__(self,
                 users: np.ndarray, items: np.ndarray, ratings: np.ndarray,
                 auto_r=False, auto_prep=True, auto_s=True, auto_pt=True):

        self.raw_data = UIRData.uir_type(users=np.array(users),
                                         items=np.array(items),
                                         ratings=np.array(ratings,
                                                          dtype=np.float))
        self.uir_n = len(users)
        self.n, self.m = 0, 0

        self.unique_values = None  # type: Optional[UIRData.uir_type]
        self.indexed_data = None  # type: Optional[UIRData.uir_type]

        self.train_index = None  # type: Optional[np.ndarray]
        self.validate_index = None  # type: Optional[np.ndarray]
        self.test_indexes = None  # type: Optional[np.ndarray]

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

        if auto_pt:
            self.preprocess_train()

    @staticmethod
    def from_csv(file_name: str) -> 'UIRData':
        """
        Load csv data from file_name. The first three columns are assumed
        to be user ids, item ids and ratings.
        :param file_name: location of the csv file
        :return: a new URIData object initialized with data from csv
        """
        df = pd.read_csv(file_name)
        return UIRData(df.values[:, 0], df.values[:, 1], df.values[:, 2])

    def preprocess(self):
        """
        Prepare some metadata for fester data manipulation.
        """
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

    def preprocess_train(self):
        """
        Calculate average ratings for user, movies and globally for the
        training dataset. For users/items not in traning dataset we use nan.
        """
        self.__needs_preprocess()

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

    def reduce(self, items=False, min_ratings=50):
        """
        Remove users/items that don't have at least min_ratings
        number of ratings.
        """
        col = 1 if items else 0
        u, c = np.unique(self.raw_data[col], return_counts=True)
        subset = np.isin(self.raw_data[col], u[c > min_ratings])
        self.raw_data = UIRData.uir_type(users=self.raw_data.users[subset],
                                         items=self.raw_data.items[subset],
                                         ratings=self.raw_data.ratings[subset])

    def split(self, train=0.6, validate=0.1, test=0.3, keep_order=False):
        """
        Prepare the split of data into train, validation and test sets.
        If keep_order then data is split consecutively.
        :param train: proportion of train data
        :param validate: proportion of validation data
        :param test: proportion of test data
        :param keep_order: keep order of input data
        """
        train, validate, test = np.array([train, validate, test]) / \
                                (train + validate + test)

        indexes = np.arange(self.uir_n)
        if not keep_order:
            np.random.shuffle(indexes)

        self.train_index, self.validate_index, self.test_indexes = \
            np.split(indexes, [int(train * self.uir_n),
                               int((train + validate) * self.uir_n)])

    def __needs_preprocess(self):
        if self.indexed_data is None:
            self.preprocess()

    @property
    def train_data(self) -> uir_type:
        self.__needs_preprocess()
        return self.uir_type(users=self.indexed_data.users[self.train_index],
                             items=self.indexed_data.items[self.train_index],
                             ratings=self.raw_data.ratings[self.train_index])

    @property
    def validate_data(self) -> uir_type:
        self.__needs_preprocess()
        return self.uir_type(users=self.indexed_data.users[self.validate_index],
                             items=self.indexed_data.items[self.validate_index],
                             ratings=self.raw_data.ratings[self.validate_index])

    @property
    def test_data(self) -> uir_type:
        self.__needs_preprocess()
        return self.uir_type(users=self.indexed_data.users[self.test_indexes],
                             items=self.indexed_data.items[self.test_indexes],
                             ratings=self.raw_data.ratings[self.test_indexes])

    def __needs_preprocess_train(self):
        if self.user_avg is None:
            self.preprocess_train()

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


if __name__ == '__main__':
    movie_lens = "../data/MovieLens/ml-latest/ratings.csv"
    uir_data = UIRData.from_csv(movie_lens)
