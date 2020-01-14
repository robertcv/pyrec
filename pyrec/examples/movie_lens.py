from pyrec.data import UIRData
from pyrec.inventory import Inventory
from pyrec.recommender import MatrixFactorization, MostInInvRecommender
from pyrec.simulator import RandomSimulator


RATINGS_FILE = "../data/MovieLens/ml-latest-small/ratings.csv"
uir_data = UIRData.from_csv(RATINGS_FILE)

mf = MatrixFactorization.load("../models/ml-small-mf")
inv = Inventory(uir_data)
miir = MostInInvRecommender(inv)

print()
print("Simulate recommending by best score:")
sim = RandomSimulator(uir_data, mf, inv)
sim.run()
print()
print("Simulate recommending by most in inventory:")
sim = RandomSimulator(uir_data, miir, inv)
sim.run()
