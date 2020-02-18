import numpy as np

from pyrec.data import UIRData
from pyrec.inventory import Inventory, UniformInventory
from pyrec.recommender import MatrixFactorization, WeightedRecommender
from pyrec.simulator import RandomFromTopNSimulator, MultiSimulator, \
    multi_success


np.random.seed(0)

# read data from file
print("load data")
RATINGS_FILE = "../../data/MovieLens/ml-latest-small/ratings.csv"
uir_data = UIRData.from_csv(RATINGS_FILE)

# create inventory for simulations
UNIFORM = False
if UNIFORM:
    inv = UniformInventory(uir_data)
else:
    inv = Inventory(uir_data)

# fit models and create sims
print("fit models")
mf = MatrixFactorization.load("../../models/ml-small-mf")
mf.data = uir_data

sims = []
alphas = [0, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1]
for a in alphas:
    _inv = inv.copy()
    wr = WeightedRecommender(_inv, mf, alpha=a)
    wr.fit(uir_data)
    sims.append(RandomFromTopNSimulator(f"a={wr.alpha}", uir_data, wr, _inv))

# run simulations
print("run simulations")
ms = MultiSimulator(50_000)
ms.set_sims(sims)
ms.run_parallel()

# plot data
print("plot data")

figure_file = "../../figures/ml"
if UNIFORM:
    figure_file += "_uinv"
else:
    figure_file += "_inv"

multi_success(sims, save_file=figure_file + "_75k_success.png")
