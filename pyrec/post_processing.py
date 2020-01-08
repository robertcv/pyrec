from pyrec.recommender import BaseRecommender


class PostProcessRecommender(BaseRecommender):

    def __init__(self, rec: BaseRecommender):
        self.rec = rec
