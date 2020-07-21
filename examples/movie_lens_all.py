from pyrec.data import UIRData
from pyrec.inventory import Inventory
from pyrec.sims.rand import BestSimulator
from pyrec.parallel import MultiSimulator
from pyrec.plots import multi_success_stops, multi_success_err, plot_ratings_dist
from pyrec.recs.mf import MatrixFactorization, NNMatrixFactorization, \
    RandomMatrixFactorization, UnbiasedMatrixFactorization
from pyrec.recs.cap_mf import CapMF
from pyrec.recs.inv import MostInInvStaticRecommender, MostInInvRecommender
from pyrec.recs.weighted import WeightedRecommender


RATINGS_FILE = "../data/MovieLens/ml-latest-small/ratings.csv"
uir_data = UIRData.from_csv(RATINGS_FILE)
inv = Inventory(uir_data)
figure_file = "../figures/inv"

k = 30
n = 20_000
rec_kwargs = {"verbose": True, "max_iteration": 200, "batch_size": 100,
              "rec": UnbiasedMatrixFactorization,
              "rec_kwargs": {"max_iteration": 200, "batch_size": 100},
              "walpha": 0.9, "rec1": UnbiasedMatrixFactorization,
              "rec1_kwargs": {"max_iteration": 200, "batch_size": 100},
              "rec2": MostInInvStaticRecommender, "rec2_kwargs": {}}
sim_kwargs = {"verbose": True}

recs = [
    ("umf", UnbiasedMatrixFactorization),
    ("miis", MostInInvStaticRecommender),
    ("r", RandomMatrixFactorization),
    ("wrs", WeightedRecommender),
    ("cap", CapMF),
]

sims = []
for name, rec in recs:
    inv = Inventory(uir_data)
    rec_kwargs["inv"] = inv
    rec_kwargs["rec2_kwargs"]["inv"] = inv
    r = rec(**rec_kwargs)
    r.fit(uir_data)
    sims.append(BestSimulator(name, uir_data, r, inv))

# run simulations
print("run simulations")
ms = MultiSimulator(n)
ms.set_sims(sims)
ms.run_parallel()

multi_success_stops(sims, list(range(1_000, n + 1, 500)),
                    save_file=figure_file + "_best_20k_success_step.png")

# repeated

# rec_kwargs["rec2_kwargs"]["inv"] = inv
# rec_kwargs["inv"] = inv
# rec_kwargs["verbose"] = False
# sim_kwargs["verbose"] = False
#
# sims = []
# for name, rec in recs:
#     rs = RepeatedSimulation(name, uir_data, inv,
#                             rec, rec_kwargs,
#                             BestSimulator, sim_kwargs)
#     rs.run(n, k)
#     sims.append(rs)
#
# multi_success_err(sims, save_file=figure_file + "_best_20k_success_err.png")
