from pyrec.data import UIRData
from pyrec.inventory import Inventory
from pyrec.recommender import MatrixFactorization, MostInInvRecommender
from pyrec.simulator import RandomFromTopNSimulator


RATINGS_FILE = "../../data/MovieLens/ml-latest-small/ratings.csv"
uir_data = UIRData.from_csv(RATINGS_FILE)

mf = MatrixFactorization.load("../../models/ml-small-mf")
mf.data = uir_data
inv = Inventory(uir_data)
miir = MostInInvRecommender(inv, mf)
miir.fit(uir_data)

print()
print("Simulate recommending by best score:")
sim = RandomFromTopNSimulator(uir_data, mf, inv)
sim.run(iter=10_000)
print()
print("Simulate recommending by most in inventory:")
sim = RandomFromTopNSimulator(uir_data, miir, inv)
sim.run(iter=10_000)
