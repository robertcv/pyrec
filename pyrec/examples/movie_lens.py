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
inv = Inventory(uir_data)
# u_inv = UniformInventory(uir_data)

# fit models and create sims
print("fit models")
inv1 = inv.copy()
mf = MatrixFactorization.load("../../models/ml-small-mf")
mf.data = uir_data
sim1 = RandomFromTopNSimulator("best score", uir_data, mf, inv1)

inv2 = inv.copy()
wr1 = WeightedRecommender(inv2, mf, alpha=0.5)
wr1.fit(uir_data)
sim2 = RandomFromTopNSimulator(f"alpha={wr1.alpha}", uir_data, wr1, inv2)

inv3 = inv.copy()
wr2 = WeightedRecommender(inv3, mf, alpha=0.75)
wr2.fit(uir_data)
sim3 = RandomFromTopNSimulator(f"alpha={wr2.alpha}", uir_data, wr2, inv3)

inv4 = inv.copy()
wr3 = WeightedRecommender(inv4, mf, alpha=0.85)
wr3.fit(uir_data)
sim4 = RandomFromTopNSimulator(f"alpha={wr3.alpha}", uir_data, wr3, inv4)

inv5 = inv.copy()
miir = MostInInvRecommender(inv5)
miir.fit(uir_data)
sim5 = RandomFromTopNSimulator("most in inv", uir_data, miir, inv5)

# run simulations
print("run simulations")
ms = MultiSimulator(50_000)
ms.set_sims([sim1, sim2, sim3, sim4, sim5])
ms.run_parallel()

# plot data
print("plot data")
sim1.plot("../../figures/ml_inv_mf.png")
sim2.plot("../../figures/ml_inv_alpha_50.png")
sim3.plot("../../figures/ml_inv_alpha_75.png")
sim4.plot("../../figures/ml_inv_alpha_85.png")
sim5.plot("../../figures/ml_inv_mii.png")

RandomFromTopNSimulator.multi_plot([sim1, sim2, sim3, sim4, sim5],
                                   data="sold_items",
                                   save_file="../../figures/ml_inv_sold.png")
RandomFromTopNSimulator.multi_plot([sim1, sim2, sim3, sim4, sim5],
                                   data="empty_items",
                                   save_file="../../figures/ml_inv_empty.png")
