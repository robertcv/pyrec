import numpy as np

from pyrec.recs.base import BaseRecommender


class WeightedRecommender(BaseRecommender):
    """
    r = alpha * rec1_r + (1 - alpha) * rec2_r
    """
    def __init__(self, alpha, rec1, rec1_kwargs: dict,
                 rec2, rec2_kwargs: dict, verbose=True, **kwargs):
        """
        This accepts either an already initialized and fitted model or
        it initializes the given class with rec_kwargs and fits it in fit.
        """
        self.alpha = alpha
        self.verbose = verbose

        self.rec1 = rec1  # type: BaseRecommender
        self.rec2 = rec2  # type: BaseRecommender
        self.rec1_kwargs = rec1_kwargs
        self.rec2_kwargs = rec2_kwargs
        self.rec1_fitted = True
        self.rec2_fitted = True

        self.rec1_kwargs["verbose"] = self.verbose
        self.rec2_kwargs["verbose"] = self.verbose

        if not isinstance(self.rec1, BaseRecommender):
            self.rec1_fitted = False
            self.rec1 = self.rec1(**self.rec1_kwargs)
        if not isinstance(self.rec2, BaseRecommender):
            self.rec2_fitted = False
            self.rec2 = self.rec2(**self.rec2_kwargs)

        super().__init__()

    def fit(self, data):
        super().fit(data)
        if not self.rec1_fitted:
            self.rec1.fit(data)
        if not self.rec2_fitted:
            self.rec2.fit(data)

    def __setattr__(self, name, value):
        if name == "data":
            self.rec1.data = value
            self.rec2.data = value
        self.__dict__[name] = value

    def _predict(self, user_index: int, item_index: int) -> float:
        r = self.alpha * self.rec1._predict(user_index, item_index) + \
            (1 - self.alpha) * self.rec2._predict(user_index, item_index)
        return r

    def _predict_user(self, user_index: int) -> np.ndarray:
        ratings = self.alpha * self.rec1._predict_user(user_index) + \
                  (1 - self.alpha) * self.rec2._predict_user(user_index)
        return ratings

    def save(self, file_name):
        self.rec1.save(file_name + "_1")
        self.rec2.save(file_name + "_2")

    def load(self, file_name):
        self.rec1.load(file_name + "_1")
        self.rec2.load(file_name + "_2")

