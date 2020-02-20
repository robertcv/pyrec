from copy import deepcopy

import numpy as np

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
        test_ratings = []
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
                test_ratings.append(np.nan)

            self.user_bought_item(user, item)

            if not self.inv.is_empty(item):
                self.inv.remove_item(item)
                sold_items.append(self.inv.percent_sold() * 100)
                not_sold_items.append(not_sold_items[-1])
            else:
                sold_items.append(sold_items[-1])
                not_sold_items.append(not_sold_items[-1] + 1)
            empty_items.append(self.inv.percent_empty() * 100)

            if not self.inv.current_count():
                break

        self._print_verbose(1)

        try:
            rmse = np.sqrt(sum(ratings_diff) / len(ratings_diff))
        except ZeroDivisionError:
            rmse = self.data.raw_data.ratings.max() - \
                   self.data.raw_data.ratings.min()

        self.sim_data = {
            "empty_items": empty_items,
            "sold_items": sold_items[1:],
            "test_ratings": test_ratings,
            "predicted_ratings": predicted_ratings,
            "not_sold_items": not_sold_items[1:],
            "rmse": rmse,
            "rmse_number": len(ratings_diff),
        }
        return deepcopy(self.sim_data)

    def rmse(self):
        print(f"{self.name}: {self.sim_data['rmse']:.2f} for {self.sim_data['rmse_number']} ratings")


class TestSimulator(BaseSimulator):
    def __init__(self, name, data: UIRData, rec: BaseRecommender,
                 inv: Inventory, verbose=True):
        self.user = max((len(i), u)
                        for u, i in data.hier_test_ratings.items())[1]
        super().__init__(name + f" (U={self.user})", data, rec, inv, verbose)

    def select_item(self, user):
        not_bought = self.user_has_not_bought(user)
        pred = self.rec.predict_user(user)[not_bought]
        best_pred = np.max(pred)
        same_indexes = np.where(pred == best_pred)[0]
        item_index = np.random.choice(same_indexes, 1, replace=False)[0]
        item = self.data.unique_values.items[not_bought][item_index]
        return item, best_pred

    def select_user(self):
        return self.user

    def user_has_not_bought(self, user):
        has_not_bought = super().user_has_not_bought(user)
        if np.all(~has_not_bought):
            raise Exception("user has bought all items")
        return has_not_bought

    def run(self, n=1000):
        has_not_bought = self.user_has_not_bought(self.user)
        if n > np.sum(has_not_bought):
            n = np.sum(has_not_bought)
            print(f"To many iterations; new n={n}")
        return super().run(n)


if __name__ == '__main__':
    from pyrec.recommender import MatrixFactorization

    RATINGS_FILE = "../../data/MovieLens/ml-latest-small/ratings.csv"
    uir_data = UIRData.from_csv(RATINGS_FILE)
    inv = Inventory(uir_data)
    mf = MatrixFactorization.load_static("../../models/ml-small-mf")
    mf.data = uir_data

    sim = TestSimulator("best score", uir_data, mf, inv)
    sim.run(10_000)
