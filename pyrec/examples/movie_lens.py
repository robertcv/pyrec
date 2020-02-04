import numpy as np

from pyrec.data import UIRData
from pyrec.inventory import Inventory, UniformInventory
from pyrec.recommender import MatrixFactorization, MostInInvRecommender, \
    WeightedRecommender
from pyrec.simulator import RandomFromTopNSimulator, MultiSimulator


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
ms = MultiSimulator(50_000)
ms.set_sims([sim1, sim2, sim3, sim4, sim5, sim6, sim7])
ms.run_parallel()

# plot data
print("plot data")

figure_file = "../../figures/ml"
if UNIFORM:
    figure_file += "_uinv"
else:
    figure_file += "_inv"

# sim1.rmse()
# sim2.rmse()
# sim3.rmse()
# sim4.rmse()
# sim5.rmse()
# sim6.rmse()
# sim7.rmse()
#
# sim1.plot_items(figure_file + "_i_mf.png")
# sim2.plot_items(figure_file + "_i_alpha_90.png")
# sim3.plot_items(figure_file + "_i_alpha_85.png")
# sim4.plot_items(figure_file + "_i_alpha_80.png")
# sim5.plot_items(figure_file + "_i_alpha_75.png")
# sim6.plot_items(figure_file + "_i_alpha_70.png")
# sim7.plot_items(figure_file + "_i_mii.png")
#
sim1.plot_ratings(figure_file + "_r_mf.png")
sim2.plot_ratings(figure_file + "_r_alpha_90.png")
sim3.plot_ratings(figure_file + "_r_alpha_85.png")
sim4.plot_ratings(figure_file + "_r_alpha_80.png")
sim5.plot_ratings(figure_file + "_r_alpha_75.png")
sim6.plot_ratings(figure_file + "_r_alpha_70.png")
sim7.plot_ratings(figure_file + "_r_mii.png")
#
# RandomFromTopNSimulator.multi_plot([sim1, sim2, sim3, sim4, sim5, sim6, sim7],
#                                    data="sold_items",
#                                    save_file=figure_file + "_sold.png")
# RandomFromTopNSimulator.multi_plot([sim1, sim2, sim3, sim4, sim5, sim6, sim7],
#                                    data="empty_items",
#                                    save_file=figure_file + "_empty.png")
# RandomFromTopNSimulator.multi_plot([sim1, sim2, sim3, sim4, sim5, sim6, sim7],
#                                    data="not_sold_items",
#                                    save_file=figure_file + "_not_sold.png")
