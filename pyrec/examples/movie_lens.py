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
sim = RandomFromTopNSimulator(uir_data, mf, inv)
sim.run(iter=100_000)
sim.plot()
print()
print("Simulate recommending by most in inventory:")
sim = RandomFromTopNSimulator(uir_data, miir, inv2)
sim.run(iter=100_000)
sim.plot()
