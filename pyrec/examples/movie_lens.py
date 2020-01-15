import numpy as np

from pyrec.data import UIRData
from pyrec.inventory import Inventory
from pyrec.recommender import MatrixFactorization, MostInInvRecommender
from pyrec.simulator import RandomFromTopNSimulator


np.random.seed(0)

RATINGS_FILE = "../../data/MovieLens/ml-latest-small/ratings.csv"
uir_data = UIRData.from_csv(RATINGS_FILE)

mf = MatrixFactorization.load("../../models/ml-small-mf")
mf.data = uir_data
inv = Inventory(uir_data)
inv2 = inv.copy()
miir = MostInInvRecommender(inv2, mf)
miir.fit(uir_data)

print()
print("Simulate recommending by best score:")
sim1 = RandomFromTopNSimulator("best score", uir_data, mf, inv)
sim1.run(iter=10_000)
sim1.plot()
print()
print("Simulate recommending by most in inventory:")
sim2 = RandomFromTopNSimulator("most in inv", uir_data, miir, inv2)
sim2.run(iter=10_000)
sim2.plot()

RandomFromTopNSimulator.multi_plot([sim1, sim2], data="sold_items")
