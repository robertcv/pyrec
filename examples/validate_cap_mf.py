import numpy as np

from pyrec.data import UIRData
from pyrec.inventory import Inventory
from pyrec.recs.cap_mf import CapMF

RATINGS_FILE = "../data/MovieLens/ml-latest-small/ratings.csv"


def rmse(model: CapMF, data: UIRData):
    max_val = data.test_data.ratings.max()
    r_pred = []
    for u, i in zip(data.test_data.users, data.test_data.items):
        r_pred.append(model._predict(u, i) * max_val)
    _rmse = np.sqrt(np.mean((np.array(r_pred) - data.test_data.ratings) ** 2))
    return _rmse


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def l(x):
    return np.log(1 + np.exp(-x))


def capacity_loss(model: CapMF, data: UIRData, inv: Inventory):
    p_i = model.p_i
    cl = []
    for i in range(data.m):
        cj = inv.counts[i]
        r_ij = model._pred_vec(np.arange(data.n), np.repeat(i, data.n))
        cl.append(l(cj - np.sum(p_i * r_ij)))
    return np.mean(cl)


def overall_objective(walpha, rmse_, cl):
    return (1 - walpha) * rmse_ ** 2 + walpha * cl


uir_data = UIRData.from_csv(RATINGS_FILE)
inv = Inventory(uir_data)
mf = CapMF(inv, walpha=1, alpha=0.01, mi=0.001)
mf.fit(uir_data)

rmse_ = rmse(mf, uir_data)
print(f"RMSE: {rmse_}")
cl = capacity_loss(mf, uir_data, inv)
print(f"Capacity Loss: {cl}")
print(f"Overall Objective: {overall_objective(0, rmse_, cl)}")
