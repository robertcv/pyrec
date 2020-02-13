import numpy as np

from pyrec.data import UIRData
from pyrec.inventory import Inventory, UniformInventory
from pyrec.recommender import MatrixFactorization, MostInInvRecommender, \
    WeightedRecommender
from pyrec.simulator import RandomFromTopNSimulator, MultiSimulator,\
    plot_ratings_violin, multi_plot


np.random.seed(0)

# read data from file
print("load data")
RATINGS_FILE = "../../data/MovieLens/ml-latest-small/ratings.csv"
uir_data = UIRData.from_csv(RATINGS_FILE)
print(len(uir_data.user_avg))
print(len(uir_data.item_avg))

# create inventory for simulations
UNIFORM = True
if UNIFORM:
    inv = UniformInventory(uir_data)
else:
    inv = Inventory(uir_data)

# fit models and create sims
print("fit models")
_inv = inv.copy()
mf = MatrixFactorization.load("../../models/ml-small-mf")
mf.data = uir_data
sim1 = RandomFromTopNSimulator("best score", uir_data, mf, _inv)

_inv = inv.copy()
wr = WeightedRecommender(_inv, mf, alpha=0.9)
wr.fit(uir_data)
sim2 = RandomFromTopNSimulator(f"alpha={wr.alpha}", uir_data, wr, _inv)

_inv = inv.copy()
wr = WeightedRecommender(_inv, mf, alpha=0.85)
wr.fit(uir_data)
sim3 = RandomFromTopNSimulator(f"alpha={wr.alpha}", uir_data, wr, _inv)

_inv = inv.copy()
wr = WeightedRecommender(_inv, mf, alpha=0.8)
wr.fit(uir_data)
sim4 = RandomFromTopNSimulator(f"alpha={wr.alpha}", uir_data, wr, _inv)

_inv = inv.copy()
wr = WeightedRecommender(_inv, mf, alpha=0.75)
wr.fit(uir_data)
sim5 = RandomFromTopNSimulator(f"alpha={wr.alpha}", uir_data, wr, _inv)

_inv = inv.copy()
wr = WeightedRecommender(_inv, mf, alpha=0.7)
wr.fit(uir_data)
sim6 = RandomFromTopNSimulator(f"alpha={wr.alpha}", uir_data, wr, _inv)

_inv = inv.copy()
miir = MostInInvRecommender(_inv)
miir.fit(uir_data)
sim7 = RandomFromTopNSimulator("most in inv", uir_data, miir, _inv)

# run simulations
print("run simulations")
ms = MultiSimulator(200_000)
ms.set_sims([sim1, sim3, sim5, sim7])
ms.run_parallel()

# plot data
print("plot data")

figure_file = "../../figures/ml"
if UNIFORM:
    figure_file += "_uinv"
else:
    figure_file += "_inv"

plot_ratings_violin(sim1, save_file=figure_file + "_rviola.png")
plot_ratings_violin(sim3, save_file=figure_file + "_rviola.png")
plot_ratings_violin(sim5, save_file=figure_file + "_rviola.png")
plot_ratings_violin(sim7, save_file=figure_file + "_rviola.png")

multi_plot([sim1, sim3, sim5, sim7],
           data="sold_items",
           save_file=figure_file + "_sold.png")
multi_plot([sim1, sim3, sim5, sim7],
           data="empty_items",
           save_file=figure_file + "_empty.png")
multi_plot([sim1, sim3, sim5, sim7],
           data="not_sold_items",
           save_file=figure_file + "_not_sold.png")
