from pyrec.data import UIRData
from pyrec.inventory import Inventory
from pyrec.sims.rand import RandomSimulator
from pyrec.parallel import MultiSimulator
from pyrec.plots import multi_success_stops
from pyrec.recs.mf import MatrixFactorization, NNMatrixFactorization, \
    RandomMatrixFactorization, UnbiasedMatrixFactorization
from pyrec.recs.inv import MostInInvStaticRecommender, MostInInvRecommender
from pyrec.recs.weighted import WeightedRecommender

uir_data = UIRData.from_csv("../data/podatki/ratings_inv_norm.csv")
inv = Inventory.from_csv("../data/podatki/inv_norm.csv")
figure_file = "../figures/t"

k = 30
n = 800_000
rec_kwargs = {"verbose": True, "max_iteration": 10, "batch_size": 1000,
              "rec": MatrixFactorization,
              "rec_kwargs": {"max_iteration": 10, "batch_size": 1000},
              "walpha": 0.5, "rec1": MatrixFactorization,
              "rec1_kwargs": {"max_iteration": 10, "batch_size": 1000},
              "rec2": MostInInvRecommender, "rec2_kwargs": {}}
sim_kwargs = {"verbose": True}

recs = [
    ("rand", RandomMatrixFactorization),
    ("miis", MostInInvStaticRecommender),
    ("mii", MostInInvRecommender),
    ("umf", UnbiasedMatrixFactorization),
    ("w(a=0.5)", WeightedRecommender),
    ("mf", MatrixFactorization),
    ("nnmf", NNMatrixFactorization),
]

sims = []
for name, rec in recs:
    inv = Inventory(uir_data)
    rec_kwargs["inv"] = inv
    rec_kwargs["rec2_kwargs"]["inv"] = inv
    r = rec(**rec_kwargs)
    r.fit(uir_data)
    sims.append(RandomSimulator(name, uir_data, r, inv))

# run simulations
print("run simulations")
ms = MultiSimulator(n)
ms.set_sims(sims)
ms.run_parallel()

multi_success_stops(sims, list(range(1_000, n + 1, 1_000)),
                    save_file=figure_file + "_rand_step.svg")
