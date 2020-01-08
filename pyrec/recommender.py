import numpy as np


def reduce_unique(data, items=False, min_ratings=50):
    """
    Remove users/items that dont have at least min_ratings.
    Args:
        data (np.ndarray): of shape (N, 3)
        items (bool): if True reduce items else reduce users
        min_ratings (int): minimum number of ratings to keep the user/item

    Returns:
        np.ndarray: reduced data
    """
    col = 1 if items else 0
    u, c = np.unique(data[:, col], return_counts=True)
    select = np.isin(data[:, col], u[c > min_ratings])
    return data[select]


class BaseRecommender:
    def fit(self, data):
        """
        Fit model to data table with columns user, item, rating.
        Args:
            data (np.ndarray): of shape (N, 3)
        """
        raise NotImplementedError

    def predict(self, user, item):
        """
        Args:
            user: userid
            item: itemid
        Returns:
            float: predicted rating for pair
        """
        raise NotImplementedError

    def top_n(self, user, n=5):
        """
        Return top N ratings in ascending order.
        Args:
            user: userid
            n: number of top ratings to return
        Returns:
            np.ndarray: itemids
            np.ndarray: predicted ratings
        """
        raise NotImplementedError
