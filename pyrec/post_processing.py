import numpy as np

from pyrec.recommender import BaseRecommender
from pyrec.inventory import Inventory


class MostInInvRecommender(BaseRecommender):

    def __init__(self, rec: BaseRecommender, inv: Inventory):
        self.rec = rec
        self.inv = inv

    def top_n(self, user, n=5):
        """Return topN items by count in inventory"""
        top_n = np.argsort(self.inv.counts)[-n:]
        top_items = self.inv.items[top_n]

        pred_r = np.zeros(len(top_n))
        for i, item in enumerate(top_items):
            pred_r[i] = self.rec.predict(user, item)

        return top_items, pred_r
