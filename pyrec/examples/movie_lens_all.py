from pyrec.data import UIRData
from pyrec.inventory import Inventory
from pyrec.recommender import MatrixFactorization, WeightedRecommender
from pyrec.simulator import RandomFromTopNSimulator, multi_success
from pyrec.parallel import MultiSimulator


RATINGS_FILE = "../../data/MovieLens/ml-latest-small/ratings.csv"

# fit models and create sims
print("fit models")
sims = []
alphas = [0, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1]
for a in alphas:
    uir_data = UIRData.from_csv(RATINGS_FILE)
    inv = Inventory(uir_data)
    wr = WeightedRecommender(a, inv, MatrixFactorization,
                             {"max_iteration": 200, "batch_size": 100})
    wr.fit(uir_data)
    sims.append(RandomFromTopNSimulator(f"a={wr.alpha}", uir_data, wr, inv))

# run simulations
print("run simulations")
ms = MultiSimulator(10_000)
ms.set_sims(sims)
ms.run_parallel()

# plot data
print("plot data")

figure_file = "../../figures/ml"
figure_file += "_inv"

multi_success(sims)
