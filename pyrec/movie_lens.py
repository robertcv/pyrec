import pandas as pd

from pyrec.inventory import Inventory
from pyrec.mf import MatrixFactorization
from pyrec.post_processing import MostInInvRecommender
from pyrec.simulator import RandomSimulator


RATINGS_FILE = "../data/MovieLens/ml-latest-small/ratings.csv"

df = pd.read_csv(RATINGS_FILE)
data = df.values[:, :-1]

mf = MatrixFactorization.load("../models/ml-small-mf")
inv = Inventory(data)
miir = MostInInvRecommender(mf, inv)

print()
print("Simulate recommending by best score:")
sim = RandomSimulator(data, mf, inv)
sim.run()
print()
print("Simulate recommending by most in inventory:")
sim = RandomSimulator(data, miir, inv)
sim.run()
