import numpy as np

from pyrec.data import UIRData
from pyrec.inventory import Inventory, UniformInventory
from pyrec.recommender import MatrixFactorization, MostInInvRecommender, \
    WeightedRecommender
from pyrec.simulator import RandomFromTopNSimulator


np.random.seed(0)

# read data from file
RATINGS_FILE = "../../data/MovieLens/ml-latest-small/ratings.csv"
uir_data = UIRData.from_csv(RATINGS_FILE)

# create inventory for simulations
inv = UniformInventory(uir_data)
# inv.reduce(0.5)
inv2 = inv.copy()
inv3 = inv.copy()

# fit recommender system models
mf = MatrixFactorization.load("../../models/ml-small-mf")
mf.data = uir_data

wr = WeightedRecommender(inv2, mf, alpha=0.85)
wr.fit(uir_data)

miir = MostInInvRecommender(inv3)
miir.fit(uir_data)

# start simulations
print()
print("Simulate recommending by best score:")
sim1 = RandomFromTopNSimulator("best score", uir_data, mf, inv)
sim1.run(iter=10_000)
sim1.plot()

print()
print(f"Simulate recommending by alpha={wr.alpha}:")
sim2 = RandomFromTopNSimulator(f"alpha={wr.alpha}", uir_data, wr, inv2)
sim2.run(iter=10_000)
sim2.plot()

print()
print("Simulate recommending by most in inventory:")
sim3 = RandomFromTopNSimulator("most in inv", uir_data, miir, inv3)
sim3.run(iter=10_000)
sim3.plot()

RandomFromTopNSimulator.multi_plot([sim1, sim2, sim3], data="sold_items")
RandomFromTopNSimulator.multi_plot([sim1, sim2, sim3], data="empty_items")
