from pyrec.data import UIRData
from pyrec.inventory import Inventory
from pyrec.sims.rand import BestSimulator
from pyrec.sims.repeated import RepeatedSimulation
from pyrec.parallel import MultiSimulator
from pyrec.plots import multi_success_stops, multi_success_err, plot_ratings_dist
from pyrec.recs.mf import MatrixFactorization, NNMatrixFactorization, \
    RandomMatrixFactorization
from pyrec.recs.inv import MostInInvStaticRecommender, MostInInvRecommender
from pyrec.recs.weighted import WeightedRecommender

# uir_data = UIRData.from_csv("../data/MovieLens/ml-1m/ratings.csv")
# inv = Inventory(uir_data)
# figure_file = "../figures/ml"

uir_data = UIRData.from_csv("../data/podatki/ratings.csv")
inv = Inventory.from_csv("../data/podatki/inv.csv")
figure_file = "../figures/podatki"

k = 30
n = 200_000
rec_kwargs = {"verbose": True, "max_iteration": 200, "batch_size": 100,
              "rec": MatrixFactorization,
              "rec_kwargs": {"max_iteration": 200, "batch_size": 100},
              "walpha": 0.7, "rec1": MatrixFactorization,
              "rec1_kwargs": {"max_iteration": 200, "batch_size": 100},
              "rec2": MostInInvRecommender, "rec2_kwargs": {}}
sim_kwargs = {"verbose": True}

recs = [
    ("wrs a=0", WeightedRecommender, 0),
    ("wrs a=0.2", WeightedRecommender, 0.2),
    ("wrs a=0.4", WeightedRecommender, 0.4),
    ("wrs a=0.6", WeightedRecommender, 0.6),
    ("wrs a=0.8", WeightedRecommender, 0.8),
    ("wrs a=1", WeightedRecommender, 1),
]

sims = []
for name, rec, a in recs:
    inv = Inventory(uir_data)
    rec_kwargs["inv"] = inv
    rec_kwargs["walpha"] = a
    rec_kwargs["rec2_kwargs"]["inv"] = inv
    r = rec(**rec_kwargs)
    r.fit(uir_data)
    sims.append(BestSimulator(name, uir_data, r, inv))

# run simulations
print("run simulations")
ms = MultiSimulator(n)
ms.set_sims(sims)
ms.run_parallel()

multi_success_stops(sims, list(range(1_000, n + 1, 1_000)),
                    save_file=figure_file + "_best_w_200k_success_step.png")

# repeated

# rec_kwargs["rec2_kwargs"]["inv"] = inv
# rec_kwargs["inv"] = inv
# rec_kwargs["verbose"] = False
# sim_kwargs["verbose"] = False
#
# sims = []
# for name, rec, a in recs:
#     rec_kwargs["walpha"] = a
#     rs = RepeatedSimulation(name, uir_data, inv,
#                             rec, rec_kwargs,
#                             BestSimulator, sim_kwargs)
#     rs.run(n, k)
#     sims.append(rs)
#
# multi_success_err(sims, save_file=figure_file + "_best_w_200k_success_err.png")
