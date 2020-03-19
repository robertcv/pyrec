from itertools import combinations, chain

import numpy as np


def ranking_score(true_rating, pred_rating):
    """slower but less memory"""
    if not len(true_rating):
        return 0

    arg_sort = np.argsort(true_rating)
    true = true_rating[arg_sort]
    pred = pred_rating[arg_sort]

    n = len(true_rating) * (len(true_rating) - 1)

    indices = np.fromiter(
        chain.from_iterable(combinations(range(len(true_rating)), 2)),
        dtype=int, count=n)
    indices.shape = int(n / 2), 2

    true_smaller = true[indices[:, 0]] < true[indices[:, 1]]
    pred_smaller = pred[indices[:, 0]] < pred[indices[:, 1]]

    all_ratings = np.sum(true_smaller)
    good_ratings = np.sum(true_smaller & pred_smaller)

    if not all_ratings:
        return 0

    return good_ratings / all_ratings


def ranking_score2(true_rating, pred_rating):
    """faster but creates len(test_ratings) * len(test_ratings) sized matrix"""
    if not len(true_rating):
        return 0

    arg_sort = np.argsort(true_rating)
    true = true_rating[arg_sort]
    pred = pred_rating[arg_sort]

    true_smaller_m = true[:, None] < true
    true_smaller = true_smaller_m[np.triu_indices(len(true_rating), k=1)]

    pred_smaller_m = pred[:, None] < pred
    pred_smaller = pred_smaller_m[np.triu_indices(len(true_rating), k=1)]

    all_ratings = np.sum(true_smaller)
    good_ratings = np.sum(true_smaller & pred_smaller)

    if not all_ratings:
        return 0

    return good_ratings / all_ratings


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
