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

    def top_nth(self, user, nth):
        """
        Return the nth top rating.
        Args:
            user: userid
            nth: number of top ratings to return
        Returns:
            float: itemid
            float: predicted rating
        """
        raise NotImplementedError
