from itertools import combinations, chain

import numpy as np


def ranking_score(true_rating, pred_rating):
    """slower but less memory"""
    if len(true_rating) == 0:
        return 0
    elif len(true_rating) == 1:
        return 1

    arg_sort = np.argsort(pred_rating)
    true = true_rating[arg_sort]

    n = len(true_rating) * (len(true_rating) - 1)

    indices = np.fromiter(
        chain.from_iterable(combinations(range(len(true_rating)), 2)),
        dtype=int, count=n)
    indices.shape = int(n / 2), 2

    true_smaller = true[indices[:, 0]] < true[indices[:, 1]]
    good_ratings = np.sum(true_smaller)

    return good_ratings / (n / 2)


def ranking_score2(true_rating, pred_rating):
    """faster but creates len(test_ratings) * len(test_ratings) sized matrix"""
    if len(true_rating) == 0:
        return 0
    elif len(true_rating) == 1:
        return 1

    arg_sort = np.argsort(pred_rating)
    true = true_rating[arg_sort]

    true_smaller_m = true[:, None] < true
    true_smaller = true_smaller_m[np.triu_indices(len(true_rating), k=1)]

    n = len(true_rating) * (len(true_rating) - 1) / 2
    good_ratings = np.sum(true_smaller)

    return good_ratings / n


if __name__ == '__main__':
    from time import time

    t = np.random.randint(10, size=1000)
    np.random.shuffle(t)
    p = np.random.rand(1000) * 10
    np.random.shuffle(p)

    a = time()
    print(ranking_score(t, p))
    print(ranking_score2(t, p))
    print(time() - a)
